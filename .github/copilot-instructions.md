# Copilot Instructions for ANLP HW2 Codebase

## Project Overview
- This codebase implements a retrieval-augmented generation (RAG) system for factual QA about Pittsburgh and CMU, as part of CMU Advanced NLP Assignment 2.
- The pipeline includes data collection (web crawling), preprocessing (corpus building, chunking), retrieval (sparse/dense/hybrid), and answer generation.
- All core retrieval logic (chunking, BM25, dense retrieval, hybrid fusion) must be implemented directly—do not use high-level RAG frameworks (e.g., LangChain, LlamaIndex).

## Key Components & Data Flow
- **Data Collection:**
  - Use `scripts/run_pipeline.py` with a config YAML/JSON to crawl web data and build the corpus.
  - Crawler config: see `configs/` for examples; outputs to `data/`.
- **Corpus & Chunking:**
  - `build_corpus` and `build_chunks` (see `src/rag_hw2/preprocess/`) process raw HTML/PDF/text into JSONL documents and chunked segments.
  - Chunking strategies are configurable (fixed-size, overlap, etc.).
- **Retrieval:**
  - Sparse (BM25) and dense (FAISS) indices are built from chunks.
  - Hybrid retrieval combines both using configurable fusion (e.g., Reciprocal Rank Fusion, weighted sum).
  - See `src/rag_hw2/retrieval/` for index and fusion logic.
- **Answering:**
  - `scripts/answer_queries.py` loads queries and runs the RAG pipeline (retriever + reader) to produce answers.
  - Supports multiple modes: sparse, dense, hybrid, closed-book.
  - Output format must match leaderboard/test requirements (see README).
- **Evaluation:**
  - `scripts/evaluate_predictions.py` computes EM, F1, ROUGE-L using reference answers.

## Developer Workflows
- **End-to-end build:**
  - `python scripts/run_pipeline.py --config configs/your_config.yaml`
- **Answer queries:**
  - `python scripts/answer_queries.py --queries leaderboard_queries.json --output preds.json [other options]`
- **Evaluate predictions:**
  - `python scripts/evaluate_predictions.py --predictions preds.json --references ref.json`
- **Config-driven:** Most scripts are driven by YAML/JSON config files for reproducibility.

## Project Conventions
- All code is in `src/rag_hw2/`; scripts in `scripts/` are thin wrappers.
- Data artifacts are stored in `data/` (raw, processed, indices).
- Only open-source models/libraries allowed (see README for policy).
- No closed-source APIs (OpenAI, Claude, etc.) or high-level RAG frameworks.
- Output formats for leaderboard/test must strictly follow README specs.

## Examples
- See `configs/crawl_example.yaml` for crawler config structure.
- See `scripts/run_pipeline.py` and `scripts/answer_queries.py` for main entry points.
- Retrieval fusion logic: `src/rag_hw2/retrieval/fusion.py`, `src/rag_hw2/retrieval/hybrid.py`.

## Integration Points
- Uses HuggingFace Transformers, FAISS, and BM25 implementations (rank-bm25/bm25s) for core ML functionality.
- All retrieval and chunking logic must be implemented in-house (not via external RAG frameworks).

---
For further details, see the project [README.md].
