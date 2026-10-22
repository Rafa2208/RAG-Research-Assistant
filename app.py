"""
Research Assistant — Flask backend (v3 — all bugs fixed)

Fixes:
  1. KB stuck on indexing → debug=False, _kb_lock prevents concurrent builds,
     full traceback in _kb_error, _kb_progress string for UI
  2. Duplicate detection → checks live _accepted_papers under lock (not stale copy)
  3. DELETE /api/papers/<paper_id> → remove file + rebuild KB
  4. POST /api/rebuild → manual refresh trigger from UI button
  5. /api/status exposes kb_building, kb_error, kb_progress
"""

import os, re, time, json, hashlib, base64, threading, traceback
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── App setup ─────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}

for d in ['uploads', 'metadata', 'chunks_store', 'papers']:
    os.makedirs(d, exist_ok=True)

# ── Globals ────────────────────────────────────────────────
_groq_client       = None
_embedding_model   = None
_reranker          = None
_faiss_index       = None
_bm25_index        = None
_all_chunks        = []
_metadata_registry = {}
_accepted_papers   = []
_retriever         = None
_kb_building       = False
_kb_error          = None
_kb_progress       = 'idle'
GROQ_MODEL         = 'llama-3.1-8b-instant'
_papers_lock       = threading.Lock()
_kb_lock           = threading.Lock()


# ══════════════════════════════════════════════════════════
#  LAZY LOADERS
# ══════════════════════════════════════════════════════════

def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        key = os.environ.get('GROQ_API_KEY', '')
        if not key:
            raise RuntimeError('GROQ_API_KEY not set.')
        _groq_client = Groq(api_key=key)
    return _groq_client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════
#  CORE RAG LOGIC
# ══════════════════════════════════════════════════════════

def call_llm(system_prompt, user_prompt, max_tokens=1024, max_retries=5):
    client = get_groq()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt}
                ],
                temperature=0.2, max_tokens=max_tokens
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err and 'tokens per day' not in err.lower():
                m    = re.search(r'try again in (\d+\.?\d*)s', err)
                wait = (float(m.group(1)) + 3) if m else 15 * (attempt + 1)
                print(f'  Rate-limit wait {wait:.0f}s ...')
                time.sleep(wait)
            elif 'tokens per day' in err.lower():
                raise
            else:
                raise
    return 'Unable to generate.'


def clean_table_name(name):
    if not name: return name
    name  = re.sub(r'\*+', '', name).strip()
    name  = re.sub(r'_+',  '', name).strip()
    words = name.split()
    if all(w == w.upper() or not w.isalpha() for w in words) and len(words) > 1:
        small = {'a','an','the','and','but','or','for','nor','on','at','to','by','in','of','up','as','is'}
        name  = ' '.join(
            w.capitalize() if (j == 0 or w.lower() not in small) else w.lower()
            for j, w in enumerate(words) if w.isalpha() or not w.isalpha()
        )
    name = re.split(r'\s{3,}', name)[0]
    for p in [r'\s+No\.\s+of\b', r'\s+Percent\b', r'\s+Respondents\s*$',
              r'\s+Reason\s+for\b', r'\s+N\s+%', r'\s+Source\b',
              r'\s+Variables\s*$', r'\s+Characteristics\b']:
        m = re.search(p, name, re.IGNORECASE)
        if m: name = name[:m.start()].strip()
    return name.strip().rstrip('.,')


def detect_pdf_type(filepath):
    import fitz
    doc          = fitz.open(filepath)
    total_pages  = len(doc)
    sample_pages = min(6, total_pages)
    digital_pages = scanned_pages = ocr_pages = 0
    for page_num in range(sample_pages):
        page      = doc[page_num]
        text      = page.get_text().strip()
        images    = page.get_images(full=True)
        has_image = len(images) > 0
        if len(text) < 50:
            scanned_pages += 1
        else:
            total_chars  = len(text)
            noise_chars  = sum(1 for c in text if c in
                '¡¿áéíóúüñ£¥§©®°²³µ¶·¼½¾àèìòùâêîôûäëïöü'
                'ÀÈÌÒÙÂÊÎÔÛÄËÏÖÜćčšžđ†‡•…‰′″‹›«»')
            lines        = text.splitlines()
            broken_words = len(re.findall(r'\b\w{1,2}\b', text))
            broken_ratio = broken_words / max(len(text.split()), 1)
            short_lines  = sum(1 for l in lines if 0 < len(l.strip()) < 8)
            short_ratio  = short_lines / max(len(lines), 1)
            noise_ratio  = noise_chars / max(total_chars, 1)
            has_full_image = False
            if has_image:
                for img in page.get_images(full=True):
                    bbox = page.get_image_bbox(img)
                    if bbox:
                        img_area  = abs(bbox.width * bbox.height)
                        page_area = page.rect.width * page.rect.height
                        if img_area > 0.5 * page_area:
                            has_full_image = True; break
            if noise_ratio > 0.015 or short_ratio > 0.35 or has_full_image:
                ocr_pages += 1
            elif broken_ratio > 0.6:
                ocr_pages += 1
            else:
                digital_pages += 1
    doc.close()
    dr    = digital_pages / sample_pages
    sr    = scanned_pages / sample_pages
    ocr_r = ocr_pages    / sample_pages
    if   sr   >= 0.7:                return 'FULLY SCANNED',  total_pages
    elif sr   >  0.25 and dr > 0.25: return 'HYBRID',         total_pages
    elif ocr_r >= 0.3:               return 'SCANNED + OCR',  total_pages
    elif dr   >= 0.6:                return 'DIGITAL BORN',   total_pages
    else:                            return 'SCANNED + OCR',  total_pages


def detect_image_pages(filepath):
    import fitz
    image_pages, doc = [], fitz.open(filepath)
    for i in range(doc.page_count):
        page   = doc[i]
        text   = page.get_text().strip()
        images = page.get_images(full=True)
        if not images: continue
        char_count = len(text)
        has_full   = any(
            abs(page.get_image_bbox(img).width * page.get_image_bbox(img).height)
            > 0.4 * page.rect.width * page.rect.height
            for img in images if page.get_image_bbox(img) is not None
        ) if images else False
        if has_full and char_count < 800:  image_pages.append(i + 1); continue
        if char_count < 200:               image_pages.append(i + 1); continue
        if char_count < 1200:
            noise = sum(1 for c in text if c in
                '¡¿£¥§©®°²³µ¶·¼½¾†‡•…‰′″àèìòùâêîôûäëïöüÀÈÌÒÙÂÊÎÔÛÄËÏÖÜćčšžđ‹›«»áéíóúüñ^~|\\<>{}[]')
            if noise / max(char_count, 1) > 0.04:
                image_pages.append(i + 1)
    doc.close()
    return image_pages


def extract_image_page_content(filepath, page_numbers):
    import fitz
    client = get_groq()
    extracted, doc = {}, fitz.open(filepath)
    for page_num in page_numbers:
        try:
            pix     = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
            img_b64 = base64.standard_b64encode(pix.tobytes('png')).decode()
            response = client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                    {'type': 'text', 'text': (
                        'Extract ALL content from this academic paper page: '
                        'table title, column headers, row labels, data values, footnotes. '
                        'Format tables as: TABLE TITLE: <title>\n| Col1 | Col2 |\n| val | val |\n'
                        'FOOTNOTES: <notes>\nReturn only extracted content.'
                    )}
                ]}],
                max_tokens=1500
            )
            extracted[page_num] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f'  Image page {page_num} failed: {e}')
            extracted[page_num] = ''
    doc.close()
    return extracted


def find_title_page(doc):
    combined = ''
    for i in range(min(5, doc.page_count)):
        t = doc[i].get_text().strip()
        if t: combined += f'\n--- PAGE {i+1} ---\n{t}\n'
    return combined


def ocr_extract_headings(all_pages_text):
    headings, seen, lines, line_nums = [], set(), all_pages_text.splitlines(), {}
    patterns = [
        re.compile(r'^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+[A-Z][A-Z\s\-,]{3,100}$'),
        re.compile(r'^(n|m|H)\.\s+[A-Z][A-Z\s\-,]{3,100}$', re.IGNORECASE),
        re.compile(r'^\d+\.\s+[A-Z][a-zA-Z\s]{3,100}$'),
    ]
    split_start = re.compile(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|H)\.\s+[A-Z][A-Z\s\-,]{2,40}$', re.IGNORECASE)
    allcaps_pat = re.compile(r'^[A-Z]{3,20}(\s[A-Z]{3,20})?$')
    skip_frags  = ['JSTOR','DOWNLOADED','STABLE URL','THIS CONTENT','ALL USE']
    known_caps  = ['ABSTRACT','INTRODUCTION','CONCLUSION','CONCLUSIONS','METHODOLOGY',
                   'METHODS','RESULTS','DISCUSSION','BACKGROUND','FINDINGS',
                   'RECOMMENDATIONS','SUMMARY','REFERENCES','APPENDIX','COMMENTS',
                   'OVERVIEW','FOOTNOTES','FORM','CONTENT']

    def add_h(h, idx):
        if h not in seen:
            headings.append(h); seen.add(h); line_nums[h] = idx

    i = 0
    while i < len(lines):
        s = lines[i].strip(); cur = i; i += 1
        if not s or len(s) < 3 or len(s) > 100: continue
        if s in seen: continue
        if any(f.lower() in s.lower() for f in skip_frags): continue
        if sum(c.isdigit() for c in s) > 4: continue
        if s.endswith(','): continue
        words = s.split()
        if len(words) > 4:
            lc = sum(1 for w in words[1:-1]
                     if w[0].islower() and w not in ['and','or','of','in','the','a','for'])
            if lc > 1: continue
        matched = False
        if split_start.match(s):
            j = i
            while j < len(lines) and not lines[j].strip(): j += 1
            if j < len(lines):
                ns = lines[j].strip()
                if (re.match(r'^[A-Z][A-Z\s\-,]{3,60}$', ns)
                        and not re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|H)\.', ns, re.IGNORECASE)):
                    combined = s + ' ' + ns
                    if combined.upper().startswith('H. '): combined = 'II.' + combined[2:]
                    if len(combined) <= 130:
                        add_h(combined, cur); seen.add(s); i = j + 1; matched = True
        if not matched:
            for pat in patterns:
                if pat.match(s):
                    norm = s
                    if s.upper().startswith('H. '):   norm = 'II.'  + s[2:]
                    elif s.lower().startswith('n. '): norm = 'II.'  + s[2:]
                    elif s.lower().startswith('m. '): norm = 'III.' + s[2:]
                    add_h(norm, cur); matched = True; break
        if not matched and allcaps_pat.match(s):
            if any(k in s for k in known_caps): add_h(s.capitalize(), cur)
        if not matched and (re.match(r'^Comments\s+on\s*$', s, re.IGNORECASE)
                            or re.match(r'^Comments$', s, re.IGNORECASE)):
            if 'COMMENTS' not in seen: add_h('COMMENTS', cur)

    headings.sort(key=lambda h: line_nums.get(h, 999))
    return headings


def ocr_extract_table_names(all_pages_text, image_content=None):
    table_names = {}

    def _parse(line_list):
        for i, line in enumerate(line_list):
            s  = line.strip()
            m1 = re.match(r'^TABLE\s+TITLE:\s+Table\s+(\d+)\s*[:\-\u2014]\s*(.+)$', s, re.IGNORECASE)
            m3 = re.match(r'^TABLE\s+(\d+)\s*[:\-\u2014]\s*(.+)$',                  s, re.IGNORECASE)
            mc = re.match(r'^TABLE\s+(\d+)\s*[-\u2014]\s*\(?\s*CONTINUED',           s, re.IGNORECASE)
            ms = re.match(r'^Table\s+(\d+)\.?\s*$',                                  s, re.IGNORECASE)
            mi = re.match(r'^Table\s+(\d+)\s*[:\-\u2014]\s*(.+)$',                  s, re.IGNORECASE)
            if m1:
                num, cap = int(m1.group(1)), m1.group(2).strip()
                if 'continued' not in cap.lower():
                    cap = clean_table_name(cap)
                    if num not in table_names and len(cap) > 3: table_names[num] = cap
            elif m3:
                num, cap = int(m3.group(1)), m3.group(2).strip()
                if 'continued' not in cap.lower():
                    cap = clean_table_name(cap)
                    if num not in table_names and len(cap) > 3: table_names[num] = cap
            elif mc:
                continue
            elif ms:
                num = int(ms.group(1)); parts = []
                for j in range(i + 1, min(i + 6, len(line_list))):
                    nxt = line_list[j].strip()
                    if not nxt: continue
                    if re.match(r'^Table\s+\d+', nxt, re.IGNORECASE): break
                    if sum(c.isdigit() for c in nxt) > 8 or '|' in nxt: break
                    parts.append(nxt)
                    if len(' '.join(parts)) > 60: break
                if parts:
                    cap = ' '.join(parts).strip()
                    if 'continued' not in cap.lower() and num not in table_names:
                        table_names[num] = clean_table_name(cap)
            elif mi:
                num, cap = int(mi.group(1)), mi.group(2).strip()
                if 'continued' not in cap.lower():
                    cap = clean_table_name(cap)
                    if num not in table_names and len(cap) > 3: table_names[num] = cap

    _parse(all_pages_text.splitlines())
    if image_content:
        for page_text in image_content.values():
            clean_lines = [
                l for l in page_text.splitlines()
                if not re.match(r'^FOOTNOTES?:', l.strip(), re.IGNORECASE)
                and 'content downloaded from' not in l.lower()
            ]
            _parse(clean_lines)
    return table_names


def ocr_extract_title_authors(title_text):
    title = 'Unknown'; authors = 'Unknown'
    lines = title_text.splitlines()
    for line in lines:
        s = line.strip()
        if s.lower().startswith('author(s):'):
            raw = s[len('author(s):'):].strip()
            if ' and ' in raw:
                parts     = re.split(r'\s+and\s+', raw)
                all_names = [x.strip() for p in parts for x in p.split(',') if x.strip()]
            else:
                all_names = [x.strip() for x in raw.split(',') if x.strip()]
            if re.search(r'\[with\s+(comments|discussion)', title_text, re.IGNORECASE):
                if len(all_names) > 1: all_names = all_names[:-1]
            authors = ', '.join(all_names); break
    page1 = ''; in_p1 = False
    for line in lines:
        if '--- PAGE 1 ---' in line: in_p1 = True; continue
        if '--- PAGE 2 ---' in line: break
        if in_p1: page1 += line + '\n'
    cands = []
    for line in page1.splitlines():
        s = line.strip()
        if not s: continue
        if s.lower().startswith('author(s):'): break
        if len(s) > 15: cands.append(s)
    if cands:
        raw_title = ' '.join(cands[:3])
        raw_title = re.sub(r'\s*\[with\s+(comments|discussion)[^\]]*\]', '', raw_title,
                           flags=re.IGNORECASE).strip()
        title = raw_title
    return title, authors


def process_ocr_paper(filepath, all_pages_text, title_text, filename):
    print(f'  [OCR] {filename}')
    table_pat   = re.compile(r'\bTable\s+(\d+)\b', re.IGNORECASE)
    table_count = len({int(m.group(1)) for m in table_pat.finditer(all_pages_text)})
    chart_pat   = re.compile(r'\b(Figure|Fig\.?|Chart)\s+(\d+)\b', re.IGNORECASE)
    chart_count = len({int(m.group(2)) for m in chart_pat.finditer(all_pages_text)})
    headings    = ocr_extract_headings(all_pages_text)
    image_pages = detect_image_pages(filepath)
    image_content = {}
    if image_pages:
        image_content = extract_image_page_content(filepath, image_pages)
    table_names = ocr_extract_table_names(all_pages_text, image_content)
    table_names = {num: clean_table_name(name) for num, name in table_names.items()}

    raw = call_llm(
        'Extract the title and the list of authors of this academic paper. '
        'Return ONLY two lines: "TITLE: <title>" and "AUTHORS: <list of authors>". '
        'Do NOT include organizations. If no authors listed, state "AUTHORS: Unknown".',
        f'TEXT:\n{title_text[:4000]}', max_tokens=200
    )
    llm_title = llm_authors = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('TITLE:'):    llm_title   = line.replace('TITLE:', '').strip()
        elif line.startswith('AUTHORS:'): llm_authors = line.replace('AUTHORS:', '').strip()
    title   = llm_title   if llm_title   and llm_title   != 'Unknown' and len(llm_title)   > 5 else 'Unknown'
    authors = llm_authors if llm_authors and llm_authors != 'Unknown' and len(llm_authors) > 3 else 'Unknown'
    if title == 'Unknown' or authors == 'Unknown':
        rt, ra = ocr_extract_title_authors(title_text)
        if title   == 'Unknown': title   = rt
        if authors == 'Unknown': authors = ra

    return dict(title=title, authors=authors, headings=headings, tables=table_count,
                table_names=table_names, charts=chart_count, chart_names={},
                pages=0, image_pages=image_pages, image_content=image_content)


def process_digital_paper(filepath, all_pages_text, title_text, filename):
    print(f'  [Digital] {filename}')
    llamaindex_text = ''
    try:
        from llama_index.core import SimpleDirectoryReader
        reader    = SimpleDirectoryReader(input_files=[filepath])
        documents = reader.load_data()
        if documents:
            llamaindex_text = '\n\n'.join([d.text for d in documents])
    except Exception as e:
        print(f'  LlamaIndex fallback: {e}')

    parse_text  = llamaindex_text if len(llamaindex_text) > len(all_pages_text) * 0.9 else all_pages_text
    table_pat   = re.compile(r'\bTable\s*(\d+)\b', re.IGNORECASE)
    table_count = len({int(m.group(1)) for m in table_pat.finditer(parse_text)})
    chart_pat   = re.compile(r'\b(Figure|Fig\.?|Chart|Graph)\s*[-]?\s*(\d+)\b', re.IGNORECASE)
    chart_count = len({int(m.group(2)) for m in chart_pat.finditer(parse_text)})
    image_pages   = detect_image_pages(filepath)
    image_content = {}

    raw = call_llm(
        'Extract the title and the list of authors of this academic paper. '
        'Return ONLY two lines: "TITLE: <title>" and "AUTHORS: <list of authors>". '
        'Do NOT include organizations. If no authors listed, state "AUTHORS: Unknown".',
        f'TEXT:\n{title_text[:4000]}', max_tokens=200
    )
    llm_title = llm_authors = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('TITLE:'):    llm_title   = line.replace('TITLE:', '').strip()
        elif line.startswith('AUTHORS:'): llm_authors = line.replace('AUTHORS:', '').strip()
    title   = llm_title   if llm_title   and llm_title   != 'Unknown' and len(llm_title)   > 5 else 'Unknown'
    authors = llm_authors if llm_authors and llm_authors != 'Unknown' and len(llm_authors) > 3 else 'Unknown'
    if title == 'Unknown' or authors == 'Unknown':
        rt, ra = ocr_extract_title_authors(title_text)
        if title   == 'Unknown': title   = rt
        if authors == 'Unknown': authors = ra

    sample = parse_text[:8000]
    headings_resp = call_llm(
        'Extract all main section headings and subheadings from the academic paper text. '
        'List them in order, each separated by a pipe symbol "|". '
        'Exclude page numbers, footnotes. '
        'Return ONLY one line: "HEADINGS: <heading1> | <heading2> | ...".',
        f'PAPER TEXT:\n{sample}', max_tokens=800
    )
    headings = []
    for line in headings_resp.splitlines():
        if line.startswith('HEADINGS:'):
            headings = [h.strip() for h in line.replace('HEADINGS:', '').split('|') if h.strip()]

    table_names = ocr_extract_table_names(parse_text, image_content)
    table_names = {num: clean_table_name(name) for num, name in table_names.items()}

    return dict(title=title, authors=authors, headings=headings, tables=table_count,
                table_names=table_names, charts=chart_count, chart_names={},
                pages=0, image_pages=image_pages, image_content=image_content)


def extract_metadata(filepath, pdf_type):
    import fitz
    doc            = fitz.open(filepath)
    total_pages    = len(doc)
    all_pages_text = ''.join(f'\n--- PAGE {i+1} ---\n{doc[i].get_text()}' for i in range(total_pages))
    title_text     = find_title_page(doc)
    doc.close()
    filename = os.path.basename(filepath)
    result   = (process_ocr_paper(filepath, all_pages_text, title_text, filename)
                if pdf_type == 'SCANNED + OCR' else
                process_digital_paper(filepath, all_pages_text, title_text, filename))
    result['pages'] = total_pages
    if result.get('title') == 'Unknown':
        base_name = os.path.splitext(filename)[0]
        cleaned   = base_name.replace('_', ' ').replace('-', ' ').strip()
        result['title'] = cleaned if len(cleaned) > 5 else f'Document from {base_name}'
    return result


# ══════════════════════════════════════════════════════════
#  KNOWLEDGE BASE BUILD
# ══════════════════════════════════════════════════════════

def rebuild_knowledge_base():
    global _faiss_index, _bm25_index, _all_chunks, _metadata_registry
    global _retriever, _kb_building, _kb_error, _kb_progress

    # Only one build at a time
    if not _kb_lock.acquire(blocking=False):
        print('[KB] Build already running, skipping.')
        return

    _kb_building = True
    _kb_error    = None
    _kb_progress = 'Starting...'

    try:
        import fitz, faiss
        from rank_bm25 import BM25Okapi

        with _papers_lock:
            papers_snapshot = list(_accepted_papers)

        if not papers_snapshot:
            _kb_progress = 'No papers.'
            return

        # Registry
        _kb_progress = 'Building registry...'
        registry = {}
        for idx, paper in enumerate(papers_snapshot):
            paper_id = paper.get('paper_id', f'paper_{str(idx+1).zfill(3)}')
            meta     = paper['metadata']
            m        = re.search(r'(\d{4})', paper['filename'])
            year     = str(m.group(1)) if m and 1900 <= int(m.group(1)) <= 2100 else 'Unknown'
            registry[paper_id] = dict(
                paper_id=paper_id, filename=paper['filename'],
                filepath=paper['filepath'], title=meta.get('title', 'Unknown'),
                authors=meta.get('authors', 'Unknown'), year=year,
                pdf_type=paper['pdf_type'], pages=meta.get('pages', 0),
                tables=meta.get('tables', 0), charts=meta.get('charts', 0),
                table_names=meta.get('table_names', {}),
                chart_names=meta.get('chart_names', {}),
                headings=meta.get('headings', [])
            )
        _metadata_registry = registry

        # Chunking
        _kb_progress = 'Chunking...'
        try:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core.schema import Document as LIDocument
            use_llama = True
        except Exception:
            use_llama = False

        all_chunks    = []
        chunk_counter = 0

        for paper_id, reg in registry.items():
            paper = next((p for p in papers_snapshot if p['filename'] == reg['filename']), None)
            if not paper: continue
            _kb_progress = f'Chunking: {reg["title"][:35]}...'
            meta     = paper['metadata']
            headings = meta.get('headings', [])
            filepath = paper['filepath']

            doc      = fitz.open(filepath)
            raw_text = ''.join(doc[i].get_text() for i in range(len(doc)))
            doc.close()

            sections = {}
            if headings:
                lines  = raw_text.splitlines()
                cur_h  = 'Preamble'; buf = []
                norm_h = {h.strip().lower(): h for h in headings}
                for line in lines:
                    stripped = line.strip()
                    matched  = norm_h.get(stripped.lower())
                    if matched:
                        if buf: sections[cur_h] = sections.get(cur_h, '') + ' '.join(buf)
                        cur_h = matched; buf = []
                    elif stripped:
                        buf.append(stripped)
                if buf: sections[cur_h] = sections.get(cur_h, '') + ' '.join(buf)
            else:
                sections = {'Full Text': raw_text}

            for heading, section_text in sections.items():
                if not section_text.strip(): continue
                if use_llama:
                    splitter   = SentenceSplitter(chunk_size=512, chunk_overlap=64)
                    li_doc     = LIDocument(text=section_text)
                    nodes      = splitter.get_nodes_from_documents([li_doc])
                    sub_chunks = [n.text for n in nodes if n.text.strip()]
                else:
                    words      = section_text.split()
                    sub_chunks = []
                    for k in range(0, len(words), 450):
                        sub_chunks.append(' '.join(words[max(0, k-50):k+450]))

                for sub in sub_chunks:
                    if len(sub.split()) < 20: continue
                    cid = f'{paper_id}_chunk_{str(chunk_counter).zfill(4)}'
                    all_chunks.append(dict(
                        chunk_id=cid, paper_id=paper_id,
                        title=reg['title'], authors=reg['authors'], year=reg['year'],
                        section=heading, text=sub.strip(), source_type='text'
                    ))
                    chunk_counter += 1

            for page_num, vis_text in meta.get('image_content', {}).items():
                if not vis_text.strip(): continue
                cid = f'{paper_id}_chunk_{str(chunk_counter).zfill(4)}'
                all_chunks.append(dict(
                    chunk_id=cid, paper_id=paper_id,
                    title=reg['title'], authors=reg['authors'], year=reg['year'],
                    section=f'Visual content - page {page_num}',
                    text=vis_text.strip(), source_type='visual'
                ))
                chunk_counter += 1

        _all_chunks = all_chunks
        print(f'[KB] {len(all_chunks)} chunks')

        if not all_chunks:
            _kb_error    = 'No text extracted from PDFs.'
            _kb_progress = 'Error: no text.'
            return

        # Embedding
        _kb_progress = f'Embedding {len(all_chunks)} chunks (may take 1-2 min)...'
        print('[KB] Loading embedding model...')
        embed_model = get_embedding_model()
        print('[KB] Encoding...')
        texts      = [c['text'] for c in all_chunks]
        embeddings = embed_model.encode(texts, batch_size=32,
                                        show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        _faiss_index = index
        print(f'[KB] FAISS: {index.ntotal} vectors')

        # BM25
        _kb_progress = 'Building BM25 index...'
        corpus      = [c['text'].lower().split() for c in all_chunks]
        _bm25_index = BM25Okapi(corpus)

        # Retriever
        _retriever   = HybridRetriever(_faiss_index, _bm25_index, _all_chunks,
                                       embed_model, _metadata_registry)
        _kb_progress = f'Ready - {len(all_chunks)} chunks, {len(registry)} papers'
        print(f'[KB] Done. {len(all_chunks)} chunks.')

    except Exception:
        _kb_error    = traceback.format_exc()
        _kb_progress = 'Build failed.'
        print(f'[KB ERROR]\n{_kb_error}')
    finally:
        _kb_building = False
        _kb_lock.release()


# ══════════════════════════════════════════════════════════
#  RETRIEVER
# ══════════════════════════════════════════════════════════

def reciprocal_rank_fusion(faiss_indices, bm25_indices, k=60):
    scores = {}
    for rank, idx in enumerate(faiss_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank_chunks(query, chunks, top_k=5):
    if not chunks or len(chunks) <= top_k:
        return chunks[:top_k]
    reranker = get_reranker()
    pairs    = [(query, c['chunk']['text'][:512]) for c in chunks]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


class HybridRetriever:
    def __init__(self, faiss_idx, bm25_idx, chunks, embed_model, metadata):
        self.faiss_index     = faiss_idx
        self.bm25_index      = bm25_idx
        self.chunks          = chunks
        self.embedding_model = embed_model
        self.metadata        = metadata

    def retrieve(self, query, top_k=5, paper_id=None, use_reranker=True):
        import faiss as f_lib
        n_candidates = top_k * 4
        qv = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        f_lib.normalize_L2(qv)
        n_search = min(n_candidates, self.faiss_index.ntotal)
        _, indices = self.faiss_index.search(qv, k=n_search)
        faiss_indices = [int(i) for i in indices[0] if i >= 0]
        bm25_scores   = self.bm25_index.get_scores(query.lower().split())
        bm25_indices  = sorted(range(len(bm25_scores)),
                                key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
        merged  = reciprocal_rank_fusion(faiss_indices, bm25_indices)
        results = []
        for idx, rrf_score in merged:
            if idx >= len(self.chunks): continue
            chunk = self.chunks[idx]
            if paper_id and chunk['paper_id'] != paper_id: continue
            paper_meta = self.metadata.get(chunk['paper_id'], {})
            results.append(dict(
                chunk=chunk, rrf_score=round(rrf_score, 6),
                paper_id=chunk['paper_id'],
                paper_title=paper_meta.get('title', 'Unknown'),
                paper_authors=paper_meta.get('authors', 'Unknown'),
                paper_year=paper_meta.get('year', 'Unknown')
            ))
            if len(results) >= n_candidates: break
        if use_reranker and len(results) > top_k:
            results = rerank_chunks(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        return results


# ══════════════════════════════════════════════════════════
#  AGENTIC Q&A
# ══════════════════════════════════════════════════════════

def agentic_qa(question, paper_id=None, max_retries=2):
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Optional, Dict, Any

    class QAState(TypedDict):
        question        : str
        paper_id        : Optional[str]
        current_query   : str
        retrieved_chunks: List[Dict[str, Any]]
        sufficient      : bool
        attempt         : int
        max_retries     : int
        answer          : str
        sources_used    : List[Dict[str, Any]]

    def retrieve_node(state):
        results = _retriever.retrieve(state['current_query'], top_k=5, paper_id=state['paper_id'])
        return {**state, 'retrieved_chunks': results}

    def self_critic_node(state):
        context = '\n\n'.join(
            f'[Chunk {i+1}]\n{r["chunk"]["text"][:300]}'
            for i, r in enumerate(state['retrieved_chunks'])
        )
        verdict = call_llm(
            'You are a strict evaluator. Does the provided text contain '
            'sufficient information to answer the question? Reply ONLY: YES or NO',
            f'Question: {state["question"]}\n\nChunks:\n{context}\n\nSufficient? YES or NO only.',
            max_tokens=5
        ).strip().upper()
        return {**state, 'sufficient': 'YES' in verdict, 'attempt': state['attempt'] + 1}

    def refine_query_node(state):
        refined = call_llm(
            'You are a search query optimizer for academic research.',
            f'Original question: {state["question"]}\nWrite a better search query. Return ONLY the query.',
            max_tokens=60
        ).strip()
        return {**state, 'current_query': refined}

    def generate_answer_node(state):
        context = '\n\n'.join(
            f'[Source {i+1}]\nPaper: {r["paper_title"]} ({r["paper_year"]})\n'
            f'Section: {r["chunk"]["section"]}\nText: {r["chunk"]["text"][:400]}'
            for i, r in enumerate(state['retrieved_chunks'])
        )
        answer = call_llm(
            'You are an academic research assistant. '
            'Answer using ONLY the provided sources. Use markdown: bold key terms, bullet points. '
            'If sources are insufficient, say so.',
            f'Question: {state["question"]}\n\nSources:\n{context}\n\nAnswer:',
            max_tokens=800
        )
        return {**state, 'answer': answer, 'sources_used': state['retrieved_chunks']}

    def should_retry(state):
        if state['sufficient']:                      return 'generate'
        if state['attempt'] >= state['max_retries']: return 'generate'
        return 'refine'

    g = StateGraph(QAState)
    g.add_node('retrieve', retrieve_node)
    g.add_node('critic',   self_critic_node)
    g.add_node('refine',   refine_query_node)
    g.add_node('generate', generate_answer_node)
    g.set_entry_point('retrieve')
    g.add_edge('retrieve', 'critic')
    g.add_conditional_edges('critic', should_retry,
                             {'refine': 'refine', 'generate': 'generate'})
    g.add_edge('refine',   'retrieve')
    g.add_edge('generate', END)
    final = g.compile().invoke(dict(
        question=question, paper_id=paper_id, current_query=question,
        retrieved_chunks=[], sufficient=False, attempt=0,
        max_retries=max_retries, answer='', sources_used=[]
    ))

    seen = []; bib = []
    for r in final['sources_used']:
        pid = r.get('paper_id', '')
        if pid in seen: continue
        seen.append(pid)
        bib.append(f'{r.get("paper_authors","?")} ({r.get("paper_year","?")}). '
                   f'*{r.get("paper_title","?")}*. Section: {r.get("chunk",{}).get("section","")}.')

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    embed_model = get_embedding_model()
    sentences   = [s.strip() for s in final['answer'].split('.') if len(s.strip()) > 20]
    contexts    = [r['chunk']['text'] for r in final['sources_used'] if r.get('chunk')]
    faithfulness = 0.0
    if sentences and contexts:
        se = embed_model.encode(sentences, convert_to_numpy=True)
        ce = embed_model.encode(contexts,  convert_to_numpy=True)
        faithfulness = round(float(np.mean([np.max(cos_sim(s.reshape(1,-1), ce)[0]) for s in se])), 4)
    embs = embed_model.encode([question, final['answer']], convert_to_numpy=True)
    relevancy = round(float(cos_sim(embs[0].reshape(1,-1), embs[1].reshape(1,-1))[0][0]), 4)

    return {'answer': final['answer'], 'bibliography': bib,
            'attempts': final['attempt'], 'faithfulness': faithfulness, 'relevancy': relevancy}


# ══════════════════════════════════════════════════════════
#  LITERATURE REVIEW
# ══════════════════════════════════════════════════════════

def generate_lit_review(paper_id):
    meta  = _metadata_registry.get(paper_id, {})
    title = meta.get('title', 'Unknown')
    year  = meta.get('year',  'Unknown')
    auth  = meta.get('authors', 'Unknown')

    print(f'[LR] Generating for: {title} ({paper_id})')

    section_queries = {
        'background' : 'background context history theory',
        'arguments'  : 'main arguments claims theoretical framework',
        'evidence'   : 'evidence examples cases data findings',
        'limitations': 'limitations problems criticism weaknesses',
        'future'     : 'future research directions unresolved questions gaps'
    }

    retrieved = {}
    for s, q in section_queries.items():
        results = _retriever.retrieve(q, top_k=4, paper_id=paper_id)
        if not results:
            results = _retriever.retrieve(q, top_k=4, paper_id=None)
        retrieved[s] = results
        print(f'[LR] Section "{s}": {len(results)} chunks')

    def ctx(results):
        if not results:
            return 'No relevant content found for this section.'
        return '\n\n'.join(
            f'[Chunk {i+1}]\n{r["chunk"]["text"][:400]}'
            for i, r in enumerate(results)
        )

    section_titles_map = {
        'background' : '1. Background and Context',
        'arguments'  : '2. Key Arguments and Claims',
        'evidence'   : '3. Evidence and Examples',
        'limitations': '4. Limitations',
        'future'     : '5. Future Directions'
    }

    known_title_variants = {
        'background' : ['Background and Context', 'Background', 'Context'],
        'arguments'  : ['Key Arguments and Claims', 'Key Arguments', 'Arguments', 'Claims'],
        'evidence'   : ['Evidence and Examples', 'Evidence', 'Examples'],
        'limitations': ['Limitations'],
        'future'     : ['Future Directions', 'Future Directions and Research Gaps',
                        'Future Research Directions', 'Future Research']
    }

    def clean_prefix(text, section_key):
        if not text:
            return ''
        lines = text.splitlines()
        if not lines:
            return text.strip()
        first = lines[0].strip()
        # Strip markdown symbols to get plain text
        plain = re.sub(r'^[#*_\s]+', '', first)
        plain = re.sub(r'[#*_\s:]+$', '', plain).strip()
        for variant in known_title_variants.get(section_key, []):
            if plain.lower() == variant.lower():
                return '\n'.join(lines[1:]).strip()
        return text.strip()

    prompts = {
        'background' : 'Write the Background and Context section. 2-3 paragraphs. Only what the paper says. Do NOT include any section title or heading.',
        'arguments'  : 'Write the Key Arguments section. 2-3 paragraphs. Do NOT include any section title or heading.',
        'evidence'   : 'Write the Evidence section. 2-3 paragraphs. Do NOT include any section title or heading.',
        'limitations': 'Write the Limitations section. 2-3 paragraphs. Do NOT include any section title or heading.',
    }

    sections = {}
    for section, prompt in prompts.items():
        try:
            raw = call_llm(
                f'Academic writer. {prompt} Use ONLY provided chunks.',
                f'Paper: {title} ({year})\n\nChunks:\n{ctx(retrieved[section])}',
                max_tokens=500
            )
            sections[section] = clean_prefix(raw, section)
            print(f'[LR] Generated: {section}')
        except Exception as e:
            print(f'[LR] Failed {section}: {e}')
            sections[section] = f'Could not generate this section.'
        time.sleep(3)

    # Future directions with NONE_FOUND guard
    try:
        future_check = call_llm(
            'Find ONLY statements where the author explicitly mentions future research or gaps. '
            'If none exist return exactly: NONE_FOUND',
            f'Paper: {title}\n\nChunks:\n{ctx(retrieved["future"])}\n\nReturn statements or NONE_FOUND.',
            max_tokens=400
        )
        time.sleep(3)
        if 'NONE_FOUND' in future_check.upper():
            sections['future'] = 'The paper does not explicitly state future research directions or gaps.'
        else:
            raw_future = call_llm(
                'Write the Future Directions section from ONLY the explicit statements below. '
                '2-3 paragraphs. Do NOT include any section title or heading.',
                f'Paper: {title}\n\nStatements:\n{future_check}',
                max_tokens=400
            )
            sections['future'] = clean_prefix(raw_future, 'future')
            time.sleep(3)
    except Exception as e:
        print(f'[LR] Future failed: {e}')
        sections['future'] = 'Could not generate future directions.'

    return dict(
        paper_id=paper_id,
        title=title,
        authors=auth,
        year=year,
        sections=[
            {'num': 1, 'title': 'Background and Context',   'body': sections.get('background', '')},
            {'num': 2, 'title': 'Key Arguments and Claims', 'body': sections.get('arguments',  '')},
            {'num': 3, 'title': 'Evidence and Examples',    'body': sections.get('evidence',   '')},
            {'num': 4, 'title': 'Limitations',              'body': sections.get('limitations','')},
            {'num': 5, 'title': 'Future Directions',        'body': sections.get('future',     '')},
        ]
    )


# ══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    return jsonify({
        'groq_configured': bool(os.environ.get('GROQ_API_KEY')),
        'papers_loaded'  : len(_accepted_papers),
        'chunks_indexed' : len(_all_chunks),
        'kb_ready'       : _retriever is not None,
        'kb_building'    : _kb_building,
        'kb_error'       : _kb_error,
        'kb_progress'    : _kb_progress
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    results = []

    for f in request.files.getlist('files'):
        if not f or not allowed_file(f.filename):
            results.append({'filename': getattr(f, 'filename', '?'),
                            'status': 'rejected', 'reason': 'Not a PDF'})
            continue

        filename = secure_filename(f.filename)
        filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        f.save(filepath)

        with open(filepath, 'rb') as fh:
            file_hash = hashlib.sha256(fh.read()).hexdigest()

        # ── Live duplicate check ──────────────────────────
        with _papers_lock:
            existing_hashes    = {p['file_hash']: p['filename'] for p in _accepted_papers}
            existing_filenames = {p['filename']  for p in _accepted_papers}

        if file_hash in existing_hashes:
            os.remove(filepath)
            results.append({'filename': filename, 'status': 'duplicate',
                            'reason': f'Same content as "{existing_hashes[file_hash]}"'})
            continue

        # Filename collision
        if filename in existing_filenames:
            stem, ext = os.path.splitext(filename)
            filename  = f'{stem}_{file_hash[:6]}{ext}'
            new_path  = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename(filepath, new_path)
            filepath  = new_path

        try:
            pdf_type, total_pages = detect_pdf_type(filepath)
        except Exception as e:
            try: os.remove(filepath)
            except: pass
            results.append({'filename': filename, 'status': 'error', 'reason': str(e)})
            continue

        if pdf_type in ['FULLY SCANNED', 'HYBRID']:
            os.remove(filepath)
            results.append({'filename': filename, 'status': 'rejected',
                            'reason': f'{pdf_type} - cannot process'})
            continue

        try:
            metadata = extract_metadata(filepath, pdf_type)
        except Exception as e:
            results.append({'filename': filename, 'status': 'error',
                            'reason': f'Metadata failed: {e}'})
            continue

        with _papers_lock:
            new_paper_id = f'paper_{str(len(_accepted_papers) + 1).zfill(3)}'
            _accepted_papers.append({
                'paper_id' : new_paper_id,
                'filename' : filename, 'filepath': filepath,
                'pdf_type' : pdf_type, 'file_hash': file_hash,
                'metadata' : metadata
        })
        results.append({
            'filename': filename, 'status': 'accepted',
            'paper_id': new_paper_id,
            'pdf_type': pdf_type, 'pages': metadata['pages'],
            'title'   : metadata['title'], 'authors': metadata['authors']
        })

    if any(r['status'] == 'accepted' for r in results):
        threading.Thread(target=rebuild_knowledge_base, daemon=False).start()

    return jsonify({'results': results})


@app.route('/api/papers')
def list_papers():
    with _papers_lock:
        snapshot = list(_accepted_papers)
    papers = []
    for idx, p in enumerate(snapshot):
        pid = p.get('paper_id', f'paper_{str(idx+1).zfill(3)}')
        papers.append({
            'paper_id': pid,
            'filename': p['filename'],
            'title'   : p['metadata'].get('title',   'Unknown'),
            'authors' : p['metadata'].get('authors', 'Unknown'),
            'pages'   : p['metadata'].get('pages',   0),
            'pdf_type': p['pdf_type']
        })
    print(f'[PAPERS] Returning: {[p["paper_id"] for p in papers]}')  # ADD THIS
    print(f'[PAPERS] Registry keys: {list(_metadata_registry.keys())}')  # ADD THIS
    return jsonify({'papers': papers})


@app.route('/api/papers/<paper_id>', methods=['DELETE'])
def delete_paper(paper_id):
    global _retriever, _faiss_index, _bm25_index, _all_chunks, _metadata_registry

    with _papers_lock:
        try:
            idx = int(paper_id.split('_')[-1]) - 1
        except (ValueError, IndexError):
            return jsonify({'error': 'Invalid paper_id'}), 400

        if idx < 0 or idx >= len(_accepted_papers):
            return jsonify({'error': 'Paper not found'}), 404

        paper = _accepted_papers[idx]
        try:
            fp = paper.get('filepath', '')
            if fp and os.path.exists(fp):
                os.remove(fp)
        except Exception:
            pass
        _accepted_papers.pop(idx)

    if not _accepted_papers:
        _retriever = None; _faiss_index = None
        _bm25_index = None; _all_chunks = []; _metadata_registry = {}
        return jsonify({'status': 'deleted', 'kb_reset': True})

    threading.Thread(target=rebuild_knowledge_base, daemon=False).start()
    return jsonify({'status': 'deleted', 'kb_reset': False})


@app.route('/api/rebuild', methods=['POST'])
def force_rebuild():
    """Manual KB rebuild — triggered by the Refresh KB button."""
    with _papers_lock:
        n = len(_accepted_papers)
    if n == 0:
        return jsonify({'error': 'No papers to index.'}), 400
    threading.Thread(target=rebuild_knowledge_base, daemon=False).start()
    return jsonify({'status': 'rebuild_started', 'papers': n})


@app.route('/api/qa', methods=['POST'])
def qa():
    if _retriever is None:
        return jsonify({'error': 'Knowledge base not ready. Check the KB badge or click Refresh KB.'}), 503
    data     = request.json or {}
    question = (data.get('question') or '').strip()
    paper_id = data.get('paper_id') or None
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        return jsonify(agentic_qa(question, paper_id=paper_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/litreview/<paper_id>')
def litreview(paper_id):
    if _retriever is None:
        return jsonify({'error': 'Knowledge base not ready.'}), 503
    if paper_id not in _metadata_registry:
        print(f'[LR 404] Requested: {paper_id}')
        print(f'[LR 404] Registry has: {list(_metadata_registry.keys())}')
        return jsonify({'error': 'Paper not found'}), 404
    try:
        result = generate_lit_review(paper_id)
        if result is None:
            return jsonify({'error': 'Literature review generation returned no content.'}), 500
        return jsonify(result)
    except Exception as e:
        print(f'[LR ERROR] {traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/configure', methods=['POST'])
def configure():
    data    = request.json or {}
    api_key = (data.get('groq_api_key') or '').strip()
    if not api_key:
        return jsonify({'error': 'No API key provided'}), 400
    os.environ['GROQ_API_KEY'] = api_key
    global _groq_client
    _groq_client = None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model=GROQ_MODEL, messages=[{'role':'user','content':'Say: ok'}], max_tokens=5
        )
        return jsonify({'status': 'connected', 'model': GROQ_MODEL})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # debug=False + use_reloader=False is MANDATORY on Windows.
    # Flask's debug reloader spawns a child process that kills background threads —
    # that was why the KB was stuck on "indexing" forever.
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)