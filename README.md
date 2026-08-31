# RAGAS Evaluation Demo

A local, disposable RAG chatbot and RAGAS evaluation workflow grounded in the seven PDFs in
`grounding-documents/`. The project demonstrates the evaluation method; its scores must not be
interpreted as an evaluation of any production RAG system.

The implementation uses seven committed PDFs, superseding the six-document count in the draft PRD.
The source files are intentionally committed to this public repository based on the project owner's
confirmation that public redistribution is acceptable.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A personal OpenAI API key with billing enabled

## Environment and secrets

Create and use a project-local virtual environment:

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv sync
```

Copy the template and add the real key locally:

```powershell
Copy-Item .env.example .env
```

All OpenAI clients receive their key from `Settings`, which loads `.env`. The application ignores
inherited process environment secrets so a shell-level `OPENAI_API_KEY` cannot override the key in
the project `.env`. It does not read secrets from source files, Streamlit configuration, command
arguments, or committed files. `.env` is ignored by Git; verify that it remains untracked before
every push.

Supported `.env` values:

| Name | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | Personal OpenAI credential |
| `RAGAS_ANSWER_MODEL` | `gpt-4.1-mini` | Chatbot answer model |
| `RAGAS_EVAL_MODEL` | `gpt-4.1` | Synthetic generation and metric judge |
| `RAGAS_EMBEDDING_MODEL` | `text-embedding-3-small` | Index and relevancy embeddings |
| `RAGAS_TOP_K` | `5` | Retrieved chunks per question |
| `RAGAS_CHUNK_TOKENS` | `800` | Maximum tokens per chunk |
| `RAGAS_CHUNK_OVERLAP` | `120` | Tokens shared by adjacent chunks |

## Staged workflow

Run every command inside the activated `.venv`, or prefix it with `uv run`.

1. Build the local Chroma index. A matching corpus/configuration fingerprint reuses the index. The
   fingerprint includes the PDF bytes, chunk settings, and embedding model; interrupted builds are
   not considered reusable.

   ```powershell
   uv run ragas-demo ingest
   # Explicitly replace it when inputs or chunk settings change:
   uv run ragas-demo ingest --rebuild
   ```

2. Launch the standalone-turn Streamlit chat interface:

   ```powershell
   uv run streamlit run app.py
   ```

3. Generate the paid 12-case synthetic test set:

   ```powershell
   uv run ragas-demo generate-testset --size 12
   ```

   Review `results/testset.jsonl` before continuing. Regeneration requires `--force` so an accidental
   rerun cannot silently repeat API spend.

4. Collect chatbot answers and retrieved contexts:

   ```powershell
   uv run ragas-demo collect-answers
   # After an interrupted run:
   uv run ragas-demo collect-answers --resume
   ```

5. Score faithfulness, answer relevancy, context precision, and context recall:

   ```powershell
   uv run ragas-demo score
   # After an interrupted run:
   uv run ragas-demo score --resume
   ```

6. Produce final artifacts:

   ```powershell
   uv run ragas-demo report
   ```

   This writes `results/scores.csv`, `results/scores.json`, and `results/report.md`. Paid scoring
   checkpoints remain under ignored `.ragas-demo/checkpoints/`. Scoring and reporting reject
   missing, duplicate, or mismatched case IDs before producing finalized artifacts.

An optional live credential/index check performs exactly one chatbot query:

```powershell
uv run ragas-demo smoke --question "What safety measures are recommended?"
```

## Validation

Normal verification is offline and does not make OpenAI calls:

```powershell
uv run pytest
uv run ruff check app.py src tests
```

The committed corpus currently produces 314 non-empty pages and 578 chunks with the default
settings. Ingestion stops rather than silently indexing encrypted, unreadable, or textless PDFs.

## Troubleshooting

- **Missing `.env` or API key:** copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
- **Index collection missing:** run `uv run ragas-demo ingest` before chat, answer collection, or smoke.
- **Corpus/config changed:** run ingestion with `--rebuild`.
- **Interrupted paid stage:** rerun that stage with `--resume`; completed cases are checkpointed.
  If the test set or responses changed since the checkpoint was written, delete the stale checkpoint
  and restart that stage intentionally.
- **Existing finalized artifact:** inspect it first, then use the stage's explicit `--force` option only
  when replacement is intentional.
- **RAGAS import regression:** keep the lockfile. RAGAS 0.4.3 still imports a module removed from
  `langchain-community` 0.4.x, so this project pins the compatible 0.3.31 release and tests the import.
