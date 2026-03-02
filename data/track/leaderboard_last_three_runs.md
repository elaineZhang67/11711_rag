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
- Run 10: `49.71%` (`F1 42.09`, `Recall 41.96`, `ROUGE 39.00`, `LLM 4.032`)
- Run 11: `49.32%` (`F1 38.58`, `Recall 43.67`, `ROUGE 35.08`, `LLM 4.197`)
- Run 12: `50.81%` (`F1 37.48`, `Recall 45.97`, `ROUGE 34.27`, `LLM 4.420`)
- Run 13: `49.83%` (`F1 37.51`, `Recall 44.99`, `ROUGE 34.50`, `LLM 4.293`)
- Run 14: `48.90%` (`F1 35.86`, `Recall 44.37`, `ROUGE 32.74`, `LLM 4.306`)
- Run 15: `50.77%` (`F1 37.48`, `Recall 45.97`, `ROUGE 34.27`, `LLM 4.414`)
- Run 16: `48.86%` (`F1 41.46`, `Recall 40.31`, `ROUGE 38.99`, `LLM 3.987`)
- Run 17: `49.46%` (`F1 38.10`, `Recall 44.73`, `ROUGE 34.90`, `LLM 4.204`)
- Run 18: `49.89%` (`F1 37.99`, `Recall 44.64`, `ROUGE 34.92`, `LLM 4.280`)
- Run 19: `49.24%` (`F1 37.33`, `Recall 44.03`, `ROUGE 34.57`, `LLM 4.242`)
- Run 20: `52.19%` (`F1 40.03`, `Recall 49.61`, `ROUGE 36.17`, `LLM 4.318`)
- Run 21: `49.83%` (`F1 40.31`, `Recall 45.76`, `ROUGE 37.46`, `LLM 4.032`)
- Run 22: `50.53%` (`F1 39.39`, `Recall 47.58`, `ROUGE 35.52`, `LLM 4.185`)
- Run 23: `50.53%` (`F1 39.39`, `Recall 47.58`, `ROUGE 35.52`, `LLM 4.185`)
- Run 24: `52.20%` (`F1 40.73`, `Recall 49.38`, `ROUGE 37.34`, `LLM 4.255`)
- Run 25: `52.44%` (`F1 40.73`, `Recall 49.38`, `ROUGE 37.34`, `LLM 4.293`)
- Run 26: `51.89%` (`F1 39.73`, `Recall 48.95`, `ROUGE 36.55`, `LLM 4.293`)
- Run 27: `52.89%` (`F1 40.12`, `Recall 48.82`, `ROUGE 36.32`, `LLM 4.452`)

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

## Run 10 (Concise Facts + t=100)
- Score: `49.71%` (`F1 42.09`, `Recall 41.96`, `ROUGE 39.00`, `LLM 4.032`)
- Key changes vs Run 9:
- prompt/postprocess behavior tightened for factual questions (direct short answers)
- explanatory questions allowed up to 3 sentences
- generation length reduced (`--max-new-tokens 100`)
- Result pattern:
- highest total score so far in this report
- large F1/ROUGE improvement
- recall/LLM-judge decreased vs Run 8/9 (more concise answers, less elaboration)

## Run 11 (Fact Truncate = 2 + Source-Attribution Cleanup)
- Score: `49.32%` (`F1 38.58`, `Recall 43.67`, `ROUGE 35.08`, `LLM 4.197`)
- Key changes vs Run 10:
- factual-question postprocess truncation changed from `1 -> 2` sentences
- added cleanup to remove context/source attribution text and bracket artifacts in answers
- Result pattern:
- total score dropped slightly vs Run 10 (`49.71 -> 49.32`)
- F1/ROUGE dropped; Recall/LLM judge increased
- practical conclusion: allowing 2 sentences helps judged quality but hurts exact-overlap metrics

## Run 12 (HyDE Enabled, MQ Off)
- Score: `50.81%` (`F1 37.48`, `Recall 45.97`, `ROUGE 34.27`, `LLM 4.420`)
- Key changes vs Run 11:
- enabled HyDE retrieval (`--hyde --hyde-max-new-tokens 64`)
- multi-query disabled (no `--multi-query`)
- kept `RRF`, `top_k=3`, `fetch_k_each=120`, `rerank_fetch_k=50`, `max_new_tokens=100`
- Result pattern:
- new best total score so far
- strongest LLM judge so far
- compared to Run 10: total score improved (`49.71 -> 50.81`) with lower F1/ROUGE but stronger Recall/LLM quality

## Run 13 (QVec Dense Index Replacement)
- Score: `49.83%` (`F1 37.51`, `Recall 44.99`, `ROUGE 34.50`, `LLM 4.293`)
- Key changes vs Run 12:
- dense index switched to question-vector indexing (generated one question per chunk, embedded those instead of raw chunk text)
- retrieval kept as hybrid + RRF + reranker (`top_k=3`, `fetch_k_each=120`, `rerank_fetch_k=50`)
- Result pattern:
- lower total score vs Run 12 (`50.81 -> 49.83`)
- slight F1/ROUGE gain over Run 12 but lower Recall/LLM judge
- practical conclusion: full dense replacement by qvec did not beat HyDE-only setup in current pipeline

## Run 14 (QVec + HyDE)
- Score: `48.90%` (`F1 35.86`, `Recall 44.37`, `ROUGE 32.74`, `LLM 4.306`)
- Key changes vs Run 13:
- enabled HyDE on top of question-vector dense index (`qvec + hyde`)
- retrieval/ranking remained hybrid + RRF + reranker (`top_k=3`, `fetch_k_each=120`, `rerank_fetch_k=50`)
- Result pattern:
- drop in total score vs Run 13 (`49.83 -> 48.90`)
- overlap metrics and recall decreased; LLM judge remained relatively high
- practical conclusion: in this setup, adding HyDE on top of qvec did not help

## Run 15 (HyDE + Factual Span Snap)
- Score: `50.77%` (`F1 37.48`, `Recall 45.97`, `ROUGE 34.27`, `LLM 4.414`)
- Key changes vs Run 14:
- switched back to non-qvec dense index (qvec off), HyDE on
- enabled conservative factual span snap (`--factual-span-snap`) with child context mode
- Result pattern:
- strong recovery vs Run 14 (`48.90 -> 50.77`)
- very close to Run 12 (`50.81`) with slightly lower LLM judge
- practical conclusion: qvec remains off; HyDE remains primary gain source

## Run 16 (HyDE + Clean Best-Answer Postprocess)
- Score: `48.86%` (`F1 41.46`, `Recall 40.31`, `ROUGE 38.99`, `LLM 3.987`)
- Key changes vs Run 15:
- stronger factoid postprocess: prefer single-sentence factual answers
- uncertainty/meta cleanup: strip `context does not...`, `however`, `[From context]`, `likely`, `appears to`
- fallback behavior changed to best-effort direct answer (no `UNKNOWN` output)
- HyDE retrieval kept on with downweight (`hyde_weight=0.85`) and lower rerank breadth (`rerank_fetch_k=40`)
- Run command:
- `python scripts/answer_queries.py --run-name rrf_hyde_t100_cleanbest --queries leaderboard_queries.json --chunks data/processed/chunks_sentence.jsonl --sparse-dir data/indices/sparse_bm25_sentence --dense-dir data/indices/dense_faiss_bge_large_v15 --mode hybrid --fusion-method rrf --top-k 3 --fetch-k-each 120 --hyde --hyde-max-new-tokens 64 --hyde-weight 0.85 --reranker-model BAAI/bge-reranker-large --rerank-fetch-k 40 --reranker-device cuda:0 --reader-backend transformers --reader-model Qwen/Qwen2.5-14B-Instruct --reader-task text-generation --device 0 --max-new-tokens 100 --andrewid Venonat2 --verbose`
- Result pattern:
- F1/ROUGE improved strongly vs Run 15
- Recall/LLM judge dropped, causing lower total score
- practical conclusion: this setting is precision/overlap-oriented but less robust on recall-oriented judging

## Run 17 (query routing and confidence fallback)
- Score: `49.46%` (`F1 38.10`, `Recall 44.73`, `ROUGE 34.90`, `LLM 4.204`)

## Run 18 (RRF + HyDE + MQ3)
- Score: `49.89%` (`F1 37.99`, `Recall 44.64`, `ROUGE 34.92`, `LLM 4.280`)
- Run command:
- `python scripts/answer_queries.py --run-name rrf_hyde_mq3 --queries leaderboard_queries.json --chunks data/processed/chunks_sentence.jsonl --sparse-dir data/indices/sparse_bm25_sentence --dense-dir data/indices/dense_faiss_bge_large_v15 --mode hybrid --fusion-method rrf --top-k 3 --fetch-k-each 120 --hyde --hyde-max-new-tokens 64 --hyde-weight 1.0 --multi-query --multi-query-max 3 --reranker-model BAAI/bge-reranker-large --rerank-fetch-k 50 --reranker-max-length 512 --reranker-device cuda:0 --reader-backend transformers --reader-model Qwen/Qwen2.5-14B-Instruct --reader-task text-generation --device 0 --max-new-tokens 100 --andrewid Venonat2 --verbose`
- Key setting summary:
- hybrid retrieval with `rrf` fusion
- HyDE enabled (`hyde_weight=1.0`)
- multi-query enabled (`multi_query_max=3`)
- reranker `BAAI/bge-reranker-large` with `rerank_fetch_k=50`
- reader `Qwen/Qwen2.5-14B-Instruct`, `max_new_tokens=100`
- Result pattern:
- improved vs Run 17 on total score and LLM judge
- still below best total score (Run 12: `50.81%`)
- next direction selected: keep HyDE on, test without MQ

## Run 19 (aaa1 attempt #3)
- Score: `49.24%` (`F1 37.33`, `Recall 44.03`, `ROUGE 34.57`, `LLM 4.242`)
- Leaderboard submission id: `aaa1` (attempt `#3/10`)
- Notes:
- this run is below Run 12 baseline and below Run 18
- decision after this run: keep/stick to Run 12 config (`rrf_hyde_t100`)

## Run 20 (Prompt Structure Update: chat-style + 5-sentence cap)
- Score: `52.19%` (`F1 40.03`, `Recall 49.61`, `ROUGE 36.17`, `LLM 4.318`)
- Leaderboard submission id: `aaa1`
- Prompt structure change (explicit):
- Before Run 20: one flat instruction prompt string from `build_rag_prompt(...)` containing both behavior rules and question/context.
- After Run 20: two-role chat prompt in `qa_reader.py`:
- `system` message: global behavior policy (professional factual QA style, concise/direct answering, context-first grounding, no meta-text).
- `user` message: task payload (`Question` + retrieved `Context` + `Answer:`).
- Implementation detail:
- added `build_answer_system_prompt()` and `build_answer_user_prompt(...)`.
- in `TransformersReader.answer(...)`, prompt is built via `_format_chat_or_fallback(...)`.
- `_format_chat_or_fallback(...)` uses tokenizer `apply_chat_template(...)` when available and falls back to old flat prompt if unavailable.
- Same pattern also applied to HyDE generation (`generate_hypothesis(...)`) with `build_hyde_system_prompt()` + `build_hyde_user_prompt(...)`.
- Length-control change coupled with prompt restructure:
- system prompt answer limit changed from `3` to `5` sentences.
- postprocess truncation changed from question-type `3/2/1` to fixed cap `5`.
- Result pattern:
- new best total score so far in this report
- strong improvements over Run 12 in F1/Recall/ROUGE
- LLM judge remains high, slightly below Run 12 peak

## Run 21 (Closed-book Baseline for Report)
- Score: `49.83%` (`F1 40.31`, `Recall 45.76`, `ROUGE 37.46`, `LLM 4.032`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode closedbook` (no retrieval, no chunk/index context used)
- interpretation:
- used as report baseline for ``retrieve-and-augment vs closed-book'' analysis
- performance is competitive on overlap metrics but lower than best RAG run on total score and judged quality

## Run 22 (Sparse-only without Reranker)
- Score: `50.53%` (`F1 39.39`, `Recall 47.58`, `ROUGE 35.52`, `LLM 4.185`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode sparse`
- no reranker (`--reranker-model` omitted)
- interpretation:
- strong sparse-only baseline for report ablation
- outperforms closed-book on total score and recall
- still below best hybrid run in this report (Run 20)

## Run 23 (Sparse-only with Reranker)
- Score: `50.53%` (`F1 39.39`, `Recall 47.58`, `ROUGE 35.52`, `LLM 4.185`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode sparse`
- with reranker (`--reranker-model BAAI/bge-reranker-large`)
- interpretation:
- score is identical to sparse-only without reranker (Run 22)
- suggests reranker did not change final top contexts in this sparse setup (`top-k=3`)

## Run 24 (Dense-only without Reranker)
- Score: `52.20%` (`F1 40.73`, `Recall 49.38`, `ROUGE 37.34`, `LLM 4.255`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode dense`
- no reranker (`--reranker-model` omitted)
- interpretation:
- very strong dense-only baseline
- slightly exceeds previous best total score (Run 20: `52.19%`) while keeping high recall and overlap metrics

## Run 25 (Dense-only with Reranker)
- Score: `52.44%` (`F1 40.73`, `Recall 49.38`, `ROUGE 37.34`, `LLM 4.293`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode dense`
- with reranker (`--reranker-model BAAI/bge-reranker-large`)
- interpretation:
- best dense-only result in this report so far
- compared with Run 24, overlap metrics are unchanged while total score/LLM judge increase

## Run 26 (Dense-only with Reranker + HyDE)
- Score: `51.89%` (`F1 39.73`, `Recall 48.95`, `ROUGE 36.55`, `LLM 4.293`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode dense`
- with reranker (`--reranker-model BAAI/bge-reranker-large`)
- with HyDE (`--hyde --hyde-max-new-tokens 64`)
- interpretation:
- lower than dense-only with reranker (Run 25), indicating HyDE did not help in this dense-only setting

## Run 27 (Hybrid weighted\_rrf, dense-heavy, stronger reranker)
- Score: `52.89%` (`F1 40.12`, `Recall 48.82`, `ROUGE 36.32`, `LLM 4.452`)
- Leaderboard submission id: `aaa1`
- Mode:
- `--mode hybrid`
- `--fusion-method weighted_rrf --dense-weight 0.8 --sparse-weight 0.2`
- reranker upgraded to `BAAI/bge-reranker-v2-m3` with larger rerank candidate pool
- interpretation:
- new best overall score in this report
- strongest LLM judge so far
- confirms dense-heavy hybrid + stronger reranker is currently the best-performing direction

## Run 28 (First Fine-tune, Not Stage 2)
- Score: `53.72%` (`F1 43.47`, `Recall 46.27`, `ROUGE 39.80`, `LLM 4.414`)
- Leaderboard submission id: `aaa1`
- Setup:
- first fine-tune only (post-initial LoRA fine-tune and merged reader model)
- explicitly **not** stage-2 fine-tune
- interpretation:
- new best total score so far
- strongest overlap metrics so far (`F1` and `ROUGE`)
- confirms first-round domain fine-tuning improved factual precision/format matching

## Run 29 (Stage-2 Fine-tune)
- Score: `54.26%` (`F1 41.63`, `Recall 50.96`, `ROUGE 36.89`, `LLM 4.503`)
- Leaderboard submission id: `aaa1`
- Setup:
- stage-2 fine-tune reader (second-round LoRA training on stage-2 QA set)
- interpretation:
- new best total score so far in this tracker
- highest recall and highest LLM judge so far
- indicates stage-2 fine-tuning improved answer coverage and judged factual usefulness

## Run 30 (Stage-2 Fine-tune + Hybrid + HyDE)
- Score: `54.83%` (`F1 41.68`, `Recall 50.92`, `ROUGE 37.55`, `LLM 4.567`)
- Leaderboard submission id: `aaa1`
- Setup:
- stage-2 fine-tuned reader + hybrid retrieval
- HyDE enabled on hybrid retrieval
- interpretation:
- new best total score so far in this tracker
- best LLM judge so far
- improved total score and ROUGE versus Run 29 while keeping recall at essentially the same level

## Run 31 (Stage-2 Model, Sparse-only + Reranker)
- Score: `53.95%` (`F1 40.42`, `Recall 50.28`, `ROUGE 35.92`, `LLM 4.567`)
- Leaderboard submission id: `aaa1`
- Setup:
- stage-2 fine-tuned reader
- sparse-only retrieval with reranker enabled
- interpretation:
- strong sparse baseline with stage-2 reader
- matches best LLM judge level while trailing Run 30 on total score/F1/ROUGE

## Run 32 (Stage-2 Model, Dense-only + Reranker)
- Score: `54.67%` (`F1 42.12`, `Recall 50.52`, `ROUGE 37.51`, `LLM 4.541`)
- Leaderboard submission id: `aaa1`
- Setup:
- stage-2 fine-tuned reader
- dense-only retrieval with reranker enabled
- interpretation:
- improves over stage-2 sparse-only reranker baseline (Run 31) on total score/F1/ROUGE
- remains slightly below stage-2 hybrid+HyDE best total score (Run 30)

## Run 33 (Stage-2 Model, Dense-only + Reranker + HyDE)
- Score: `54.50%` (`F1 41.85`, `Recall 50.32`, `ROUGE 37.31`, `LLM 4.541`)
- Leaderboard submission id: `aaa1`
- Setup:
- stage-2 fine-tuned reader
- dense-only retrieval with reranker enabled
- HyDE enabled
- interpretation:
- slightly below Run 32 on total/F1/ROUGE with nearly identical recall and identical LLM judge
- in this dense-only setup, HyDE did not improve final metrics


## Model Size Summary (What Scaled Up)
- Reader model: unchanged (`Qwen2.5-14B-Instruct`, 14B class)
- Embedding model: moved from small MiniLM class to larger BGE-large class
- Reranker: moved from smaller MiniLM cross-encoder class to larger BGE-reranker-large class
- Practical effect: better candidate ranking/recall, then normalization/prompt tightening improved precision
