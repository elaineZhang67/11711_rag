# RAG Starter Runbook (HW2)

This codebase implements the assignment pipeline end-to-end without high-level RAG frameworks:

1. Scrape / collect raw data
2. Build cleaned corpus
3. Chunk documents
4. Build sparse (BM25) index
5. Build dense (FAISS + embeddings) index
6. Run sparse / dense / hybrid QA
7. Run evaluation and ablations

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Crawl (optional, for new sources)

Use the provided config example:

```bash
python3 scripts/crawl_web.py --config configs/crawl_example.yaml --verbose
```

Or run the full build pipeline with crawling:

```bash
python3 scripts/run_pipeline.py --config configs/full_pipeline_example.yaml --verbose
```

## Step 2: Build Corpus from Baseline + Crawled Data

The provided baseline dump is outside this folder (`../baseline_data`).

```bash
python3 scripts/build_corpus.py \
  --input-dir ../baseline_data \
  --input-dir data/raw/crawl_visitpgh_cmu \
  --output data/processed/documents.jsonl \
  --verbose
```

## Step 3: Chunk Documents

Fixed-size baseline:

```bash
python3 scripts/build_chunks.py \
  --documents data/processed/documents.jsonl \
  --output data/processed/chunks.jsonl \
  --strategy fixed \
  --chunk-size-words 250 \
  --overlap-words 50
```

Sentence-aware variant (for ablation):

```bash
python3 scripts/build_chunks.py \
  --documents data/processed/documents.jsonl \
  --output data/processed/chunks_sentence.jsonl \
  --strategy sentence \
  --chunk-size-words 250 \
  --overlap-words 50
```

## Step 4: Build Sparse Index (BM25)

```bash
python3 scripts/build_sparse_index.py \
  --chunks data/processed/chunks.jsonl \
  --out-dir data/indices/sparse_bm25
```

## Step 5: Build Dense Index (Embeddings + FAISS)

```bash
python3 scripts/build_dense_index.py \
  --chunks data/processed/chunks.jsonl \
  --out-dir data/indices/dense_faiss \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --verbose
```

## Step 6: Answer Leaderboard Queries

With an open HuggingFace model (example):

```bash
python3 scripts/answer_queries.py \
  --queries leaderboard_queries.json \
  --output data/outputs/leaderboard_flan_t5_base.json \
  --debug-output data/outputs/leaderboard_flan_t5_base.debug.jsonl \
  --chunks data/processed/chunks.jsonl \
  --sparse-dir data/indices/sparse_bm25 \
  --dense-dir data/indices/dense_faiss \
  --mode hybrid \
  --fusion-method rrf \
  --reader-backend transformers \
  --reader-model google/flan-t5-base \
  --reader-task text2text-generation \
  --max-new-tokens 32 \
  --andrewid YOUR_ANDREWID \
  --verbose
```

## Step 7: Ablations for Report

This runs sparse-only, dense-only, and hybrid variants and stores predictions/debug traces.

```bash
python3 scripts/run_ablation.py \
  --queries leaderboard_queries.json \
  --out-dir data/outputs/ablations \
  --chunks data/processed/chunks.jsonl \
  --sparse-dir data/indices/sparse_bm25 \
  --dense-dir data/indices/dense_faiss \
  --reader-backend transformers \
  --reader-model google/flan-t5-base \
  --reader-task text2text-generation
```

If you have your own labeled dev references:

```bash
python3 scripts/run_ablation.py \
  --queries data/dev/questions.json \
  --references data/dev/reference_answers.json \
  --reader-backend transformers \
  --reader-model google/flan-t5-base \
  --reader-task text2text-generation
```

## Step 8: Evaluate Predictions (local dev set)

```bash
python3 scripts/evaluate_predictions.py \
  --predictions data/outputs/dev_preds.json \
  --references data/dev/reference_answers.json \
  --output data/outputs/dev_scores.json
```

## Notes

- Retrieval logic is implemented manually in `src/rag_hw2/retrieval/` (BM25, dense FAISS, hybrid fusion).
- Chunking logic is implemented manually in `src/rag_hw2/chunking.py`.
- The crawler is intentionally simple and domain-whitelisted. Add more seed URLs/configs for better event coverage.
- For the final test release, run `scripts/answer_queries.py` on the released question file and produce `system_output_{1,2,3}.json`.
