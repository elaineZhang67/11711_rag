# 11711 RAG Pipeline

1. scrape raw pages/PDFs
2. build cleaned corpus
3. chunk documents
4. build sparse + dense indexes
5. answer query sets with hybrid retrieval + reranking + reader model

## 0) Environment

Recommended: Python 3.11.x

```bash
pip install -r requirements.txt
```

Run all commands from repo root:

```bash
cd anlp-spring2026-hw2-main
```

## 1) Scrape Data

### 1.1 Scrape recommended Pittsburgh sources

```bash
python scripts/scrape_recommended_sources.py \
  --out-root data/raw/recommended_sources \
  --verbose
```

### 1.2 Scrape broad CMU sources

```bash
python scripts/scrape_cmu_all.py \
  --out-dir data/raw/cmu \
  --max-pages 3000 \
  --max-depth 5 \
  --sleep-sec 0.8 \
  --verbose
```

Notes:
- `--max-depth` is clamped to 4-5 by the script.
- Scrape results are stored with `manifest.jsonl` under each output directory.

## 2) Build Corpus

Merge baseline + scraped data into one cleaned JSONL:

```bash
python scripts/build_corpus.py \
  --input-dir baseline_data \
  --input-dir data/raw/recommended_sources \
  --input-dir data/raw/cmu \
  --output data/processed/documents_v2.jsonl \
  --min-chars 100 \
  --quality-filter \
  --verbose
```

What this does:
- HTML extraction + boilerplate removal
- PDF text extraction
- text normalization + dedupe
- optional quality filtering

## 3) Build Chunks

Sentence-aware chunking (recommended):

```bash
python scripts/build_chunks.py \
  --documents data/processed/documents_v2.jsonl \
  --output data/processed/chunks_sentence_v2.jsonl \
  --strategy sentence \
  --chunk-size-words 250 \
  --overlap-words 50 \
  --min-chunk-words 20 \
  --verbose
```

## 4) Build Retrieval Indexes

### 4.1 Sparse (BM25)

```bash
python scripts/build_sparse_index.py \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --out-dir data/indices/sparse_bm25_sentence_v2
```

### 4.2 Dense (FAISS + BGE)

```bash
python scripts/build_dense_index.py \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --out-dir data/indices/dense_faiss_bge_large_v15_v2 \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --batch-size 64 \
  --verbose
```

## 5) Run QA (Hybrid + HyDE)

```bash
python scripts/answer_queries.py \
  --run-name xxx \
  --queries leaderboard_queries.json \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --sparse-dir data/indices/sparse_bm25_sentence_v2 \
  --dense-dir data/indices/dense_faiss_bge_large_v15_v2 \
  --mode hybrid \
  --fusion-method rrf \
  --top-k 3 \
  --fetch-k-each 120 \
  --hyde \
  --hyde-max-new-tokens 64 \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --rerank-fetch-k 50 \
  --reranker-max-length 512 \
  --reranker-device cuda:0 \
  --reader-backend transformers \
  --reader-model Qwen/Qwen2.5-14B-Instruct \
  --reader-task text-generation \
  --device 0 \
  --max-new-tokens 100 \
  --andrewid x \
  --verbose
```

Output is created automatically at:

- `data/outputs/<run-name>/answers.json`
- `data/outputs/<run-name>/command.txt`
- `data/outputs/<run-name>/leaderboard.result.txt`

## 6) Other Example Retrieval Modes 

### Sparse-only

```bash
python scripts/answer_queries.py \
  --run-name sparse_only \
  --queries leaderboard_queries.json \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --sparse-dir data/indices/sparse_bm25_sentence_v2 \
  --mode sparse \
  --top-k 3 \
  --reader-backend transformers \
  --reader-model Qwen/Qwen2.5-14B-Instruct \
  --reader-task text-generation \
  --device 0 \
  --max-new-tokens 100 \
  --andrewid aaa1
```

### Dense-only

```bash
python scripts/answer_queries.py \
  --run-name dense_only \
  --queries leaderboard_queries.json \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --dense-dir data/indices/dense_faiss_bge_large_v15_v2 \
  --mode dense \
  --top-k 3 \
  --reader-backend transformers \
  --reader-model Qwen/Qwen2.5-14B-Instruct \
  --reader-task text-generation \
  --device 0 \
  --max-new-tokens 100 \
  --andrewid aaa1
```

### Closed-book (no retrieval)

```bash
python scripts/answer_queries.py \
  --run-name closedbook \
  --queries leaderboard_queries.json \
  --mode closedbook \
  --reader-backend transformers \
  --reader-model Qwen/Qwen2.5-14B-Instruct \
  --reader-task text-generation \
  --device 0 \
  --max-new-tokens 100 \
  --andrewid aaa1
```

## 7) Query File Format

`answer_queries.py` expects JSON input (`list` or `dict`), not plain `.txt`.

Example dict format:

```json
{
  "1": "When was Carnegie Mellon University founded?",
  "2": "What is the CMU mascot?"
}
```

## 8) Optional: LoRA Fine-tuning

Train LoRA adapter:

```bash
python scripts/finetune_lora.py \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --questions data/train/questions.txt \
  --references data/train/reference_answers.json \
  --output-dir data/models/qwen25_14b_lora_stage1 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8
```

Merge adapter for inference:

```bash
python scripts/merge_lora.py \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --adapter-dir data/models/qwen25_14b_lora_stage1 \
  --out-dir data/models/qwen25_14b_stage1_merged \
  --dtype bf16
```

Use merged model in QA:

```bash
python scripts/answer_queries.py \
  --run-name stage1_hybrid \
  --queries leaderboard_queries.json \
  --chunks data/processed/chunks_sentence_v2.jsonl \
  --sparse-dir data/indices/sparse_bm25_sentence_v2 \
  --dense-dir data/indices/dense_faiss_bge_large_v15_v2 \
  --mode hybrid \
  --fusion-method rrf \
  --top-k 3 \
  --fetch-k-each 120 \
  --hyde \
  --hyde-max-new-tokens 64 \
  --reranker-model BAAI/bge-reranker-large \
  --rerank-fetch-k 50 \
  --reranker-device cuda:0 \
  --reader-backend transformers \
  --reader-model data/models/qwen25_14b_stage1_merged \
  --reader-task text-generation \
  --device 0 \
  --max-new-tokens 100 \
  --andrewid aaa1
```