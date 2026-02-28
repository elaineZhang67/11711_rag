# Leaderboard Runs (Exact Change Log)

## Score Trend
- Run 1: `31.46%` (`F1 22.86`, `Recall 20.29`, `ROUGE 18.85`, `LLM 3.554`)
- Run 2: `39.83%` (`F1 21.18`, `Recall 35.53`, `ROUGE 18.04`, `LLM 4.382`)
- Run 3: `46.34%` (`F1 31.26`, `Recall 45.68`, `ROUGE 28.50`, `LLM 4.197`)
- Run 4: `30.22%` (`F1 24.28`, `Recall 17.72`, `ROUGE 20.14`, `LLM 3.350`)
- Run 5: `42.10%` (`F1 32.86`, `Recall 40.09`, `ROUGE 30.32`, `LLM 3.605`)
- Run 6: `46.85%` (`F1 31.74`, `Recall 46.51`, `ROUGE 28.89`, `LLM 4.210`)
- Run 7: `49.05%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.299`)
- Run 8: `49.29%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.338`)
- Run 9: `49.25%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.331`)

## Run 1 (Baseline)
- Retrieval mode: `hybrid`
- Chunking: `sentence`, `chunk_size_words=250`, `overlap_words=50`, `min_chunk_words=20`
- Sparse retriever: BM25
- Dense retriever: FAISS + `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Retrieval config: `top_k=5`, `fetch_k_each=50`, `rerank_fetch_k=20`
- Reader model: `Qwen/Qwen2.5-14B-Instruct`
- Generation config: `max_new_tokens=32`, `temperature=0.0`
- Prompt style: short direct answer, but still general QA wording
- Normalization strategy (baseline):
- prefix strip (`Answer:`, etc.)
- first-line keep
- trim quotes / punctuation
- Result pattern: low recall and many canonical-form mismatches

## Run 2 (Recall-Oriented)
- Kept: same chunking and same reader model (`Qwen/Qwen2.5-14B-Instruct`)
- Retrieval config changes vs Run 1:
- `top_k: 5 -> 3`
- `fetch_k_each: 50 -> 80`
- `rerank_fetch_k: 20 -> 30`
- Prompt changes vs Run 1:
- relaxed to allow best-effort output when context is weak
- more permissive detail/wording instructions
- Generation changes vs Run 1:
- `max_new_tokens: 32 -> 150`
- Normalization additions vs Run 1:
- alias list collapse (`A; Full A -> A` for singular-style answers)
- parenthetical alias collapse (`A (Full A) -> A` when one contains the other)
- trailing explanation trim (`A - explanation -> A`)
- simple date/year cleanup (`in 1967 -> 1967`, and question-aware year/date handling)
- Result pattern:
- Recall and LLM judge increased a lot
- F1/ROUGE stayed low due to verbose answers / non-canonical phrasing

## Run 3 (Precision + Canonicalization)
- Kept: same chunking (`sentence`, `250/50`)
- Model changes vs Run 2:
- dense embedding: `all-MiniLM-L6-v2 -> BAAI/bge-large-en-v1.5`
- reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2 -> BAAI/bge-reranker-large`
- Retrieval/pipeline changes vs Run 2:
- `fetch_k_each: 80 -> 120`
- `rerank_fetch_k: 30 -> 50`
- added score shaping before rerank:
- boost title/url/text query-term matches
- penalty for list/directory/faculty-style noisy pages
- added extractive-first path for date/year/person-like questions, with LLM fallback
- Prompt/output changes vs Run 2:
- switched back to canonical short-answer style
- removed verbose lead-ins by instruction (`The answer is ...`)
- set concise answer behavior (keyword/span-like answers)
- Generation changes vs Run 2:
- `max_new_tokens: 150 -> 120`
- Normalization additions vs Run 2:
- citation tail removal (`[1]`)
- stronger verbose-intro stripping (`The name is ... -> ...`)
- singular-fact compression to short spans
- one-sentence final cap in postprocessing
- Result pattern:
- best overall score so far
- major F1/ROUGE gain from cleaner exact spans
- recall remained strong from broader retrieval + stronger reranking

## Run 4 (Fusion Ablation Regression)
- Score: `30.22%` (`F1 24.28`, `Recall 17.72`, `ROUGE 20.14`, `LLM 3.350`)
- Context:
- first fusion-ablation attempt after Run 3 baseline
- hybrid method/config changed from best-known baseline
- Result pattern:
- strong Recall drop (`45.68 -> 17.72`)
- Total score dropped below Run 1/2/3
- indicates this fusion setting should not be used as final default
- Action:
- keep Run 3 as anchor baseline
- continue ablation with alternative fusion settings, but revert immediately if Recall collapses

## Run 5 (Postprocessing Sentence Cap = 3)
- Score: `42.10%` (`F1 32.86`, `Recall 40.09`, `ROUGE 30.32`, `LLM 3.605`)
- Change applied:
- postprocessing sentence cap updated from `1 -> 3`
- file: `rag_hw2/reader/qa_reader.py`
- call site change: `_truncate_to_max_sentences(..., max_sentences=3)`
- Result pattern:
- large recovery vs Run 4 on total score and recall
- best F1/ROUGE seen so far in this report
- still below Run 3 total score, so retrieval/fusion settings remain the main lever

## Run 6 (RRF + MQ1)
- Score: `46.85%` (`F1 31.74`, `Recall 46.51`, `ROUGE 28.89`, `LLM 4.210`)
- Command/config:
- `--mode hybrid`
- `--fusion-method rrf`
- `--top-k 3`
- `--fetch-k-each 120`
- `--multi-query --multi-query-max 1`
- `--reranker-model BAAI/bge-reranker-large`
- `--rerank-fetch-k 50`
- `--reader-model Qwen/Qwen2.5-14B-Instruct`
- `--max-new-tokens 120`
- Result pattern:
- new best total score in this report
- recall and LLM judge both improved vs prior best
- confirms conservative MQ (`max=1`) can help when fusion stays on `rrf`

## Run 7 (RRF + MQ1 + No Extractive-First)
- Score: `49.05%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.299`)
- Key change vs Run 6:
- disabled extractive-first final answer path in `rag_hw2/pipeline.py`
- system now always uses reader generation + postprocessing for final answer
- Result pattern:
- best overall score so far
- improved across all reported metrics (F1/Recall/ROUGE/LLM judge)
- consistent with removal of brittle person/date snippet extraction errors

## Run 8 (RRF + MQ3 + No Extractive-First)
- Score: `49.29%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.338`)
- Key change vs Run 7:
- `--multi-query-max: 1 -> 3`
- Result pattern:
- marginal total-score gain (`49.05 -> 49.29`)
- F1/Recall/ROUGE unchanged
- small LLM-judge increase (`4.299 -> 4.338`)
- practical conclusion: MQ=3 is slightly better but effect size is small

## Run 9 (RRF + MQ3 + Doc Diversification)
- Score: `49.25%` (`F1 33.75`, `Recall 49.20`, `ROUGE 30.77`, `LLM 4.331`)
- Key change vs Run 8:
- enabled post-rerank doc diversification (`--diversify-docs --doc-cap 2`)
- Result pattern:
- slight regression vs Run 8 (`49.29 -> 49.25`)
- core overlap metrics unchanged; small LLM-judge drop (`4.338 -> 4.331`)
- practical conclusion: doc-cap diversification did not help at current `top_k=3`

## Model Size Summary (What Scaled Up)
- Reader model: unchanged (`Qwen2.5-14B-Instruct`, 14B class)
- Embedding model: moved from small MiniLM class to larger BGE-large class
- Reranker: moved from smaller MiniLM cross-encoder class to larger BGE-reranker-large class
- Practical effect: better candidate ranking/recall, then normalization/prompt tightening improved precision
