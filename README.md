# A RAG Based Intelligent Research Assistant for Academic Document Analysis, Automated Question Answering and Literature Review Using Hybrid Retrieval

> An agentic, open source research assistant that lets humanities and social science scholars upload their own academic PDF corpus and interact with it through grounded question answering and structured literature review generation — running entirely on free tools with no GPU required.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [PDF Ingestion and Classification](#pdf-ingestion-and-classification)
4. [Dual Path Metadata Extraction](#dual-path-metadata-extraction)
5. [Document Type Acceptance Criteria](#document-type-acceptance-criteria)
6. [Section Aware Chunking](#section-aware-chunking)
7. [Embedding Model](#embedding-model)
8. [Dual Indexing — FAISS and BM25](#dual-indexing--faiss-and-bm25)
9. [Hybrid Retrieval with Reciprocal Rank Fusion](#hybrid-retrieval-with-reciprocal-rank-fusion)
10. [CrossEncoder Reranking](#crossencoder-reranking)
11. [Agentic Q&A Mode — Mode 1](#agentic-qa-mode--mode-1-langgraph-state-graph)
12. [Literature Review Mode — Mode 2](#literature-review-mode--mode-2-hallucination-safeguard)
13. [LLM API Used](#llm-api-used)
14. [Evaluation Metrics](#evaluation-metrics)
15. [Model Response Comparison](#model-response-comparison-vs-claude)
16. [Deployment Tools and Stack](#deployment-tools-and-stack)
17. [Project Structure](#project-structure)
18. [Installation and Setup](#installation-and-setup)
19. [How to Run](#how-to-run)
20. [Environment Variables](#environment-variables)
21. [Screenshots](#screenshots)
22. [References](#references)

---

## Project Overview

This project builds an end to end Agentic Retrieval Augmented Generation pipeline deployed as a Flask web application. It accepts a user uploaded collection of academic PDFs, classifies and processes them through a dual path extraction pipeline, indexes them using both dense and sparse retrieval methods, and supports two modes of scholarly interaction.

**Mode 1 — Agentic Q&A:** A LangGraph directed state graph that retrieves, self critiques, and refines answers before generating a response.

**Mode 2 — Literature Review:** Automatically generates a structured five section academic review of any single uploaded paper with an explicit hallucination safeguard on future directions.

The system uses the Groq API for LLM inference, runs on CPU only hardware, and requires no paid subscriptions beyond the free Groq tier.

---

## System Architecture