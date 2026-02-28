# Compact Combined Version (Original Files Kept)

This folder uses a compact CLI:

- `compact/hw2_rag_compact.py`

And it now calls **combined implementation files** (no deletions):

- `compact/collect_compact.py` (crawler + source catalog)
- `compact/data_compact.py` (HTML/PDF parsing + corpus build + chunking)
- `compact/retrieval_compact.py` (chunk store + BM25 + FAISS + fusion + hybrid)
- `compact/qa_compact.py` (query I/O + reader + pipeline + eval metrics)

## Common Commands

```bash
python3 compact/hw2_rag_compact.py list-sources
python3 compact/hw2_rag_compact.py scrape --groups general events food --verbose
python3 compact/hw2_rag_compact.py build-data --input-dir baseline_data --input-dir data/raw/recommended_sources --verbose
python3 compact/hw2_rag_compact.py build-indices --verbose
python3 compact/hw2_rag_compact.py answer --queries leaderboard_queries.json --output data/outputs/leaderboard.json --mode hybrid --reader-backend heuristic
```
