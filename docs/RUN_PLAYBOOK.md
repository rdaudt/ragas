# RAGAS Demo Run Playbook

This playbook is the operator checklist for running the local app and the staged RAGAS
evaluation. Commands assume Windows PowerShell from the repository root.

## 1. Preflight

Confirm you are in the repo and on the expected branch:

```powershell
git status --short --branch
```

Confirm Python 3.12 and `uv` are available:

```powershell
python --version
uv --version
```

Create or refresh the project-local virtual environment:

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv sync
```

Create the local secrets file if it does not already exist:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to a personal key with billing enabled. The app intentionally
loads secrets from `.env`; inherited shell environment variables do not override the project file.

Before pushing code, confirm `.env` remains untracked:

```powershell
git status --short
```

## 2. Offline Verification

Run the offline test and lint checks before paid calls:

```powershell
uv run pytest
uv run ruff check app.py src tests
```

These commands do not call OpenAI.

## 3. Build The Retrieval Index

Build the local Chroma index:

```powershell
uv run ragas-demo ingest
```

Expected result with the committed PDFs and default settings:

```text
Index built: 7 documents, 314 pages, 578 chunks.
```

If the command reports `Index reused`, the stored index fingerprint matches the current PDF bytes,
chunk settings, and embedding model.

Force a rebuild after changing PDFs, chunk settings, or embedding model:

```powershell
uv run ragas-demo ingest --rebuild
```

Interrupted index builds are not reused; rerun `ingest` to replace the incomplete collection.

## 4. Run The Chat App

Start Streamlit:

```powershell
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Ask a question grounded in the PDFs. Each app question is a paid OpenAI answer-model call and uses
the local Chroma index for retrieval.

Optional one-query smoke test from the CLI:

```powershell
uv run ragas-demo smoke --question "What safety measures are recommended?"
```

## 5. Generate The RAGAS Test Set

Generate the reviewed synthetic test set:

```powershell
uv run ragas-demo generate-testset --size 12
```

This is a paid RAGAS generation step. It writes:

```text
results/testset.jsonl
```

Manually review `results/testset.jsonl` before continuing. Each row should have a plausible
`user_input` and `reference` answer for the committed corpus.

If replacement is intentional:

```powershell
uv run ragas-demo generate-testset --size 12 --force
```

After forcing a new test set, do not reuse older responses or score checkpoints.

## 6. Collect Chatbot Answers

Run the finalized test-set questions through the same chatbot service:

```powershell
uv run ragas-demo collect-answers
```

This is a paid answer-model step. It writes:

```text
results/responses.jsonl
```

If interrupted:

```powershell
uv run ragas-demo collect-answers --resume
```

Resume validates that the response checkpoint belongs to the current test set by `case_id`,
`user_input`, and `reference`. If it fails, delete the stale `results/responses.jsonl` only after
confirming you intend to restart answer collection for the current test set.

## 7. Score With RAGAS

Score the collected responses:

```powershell
uv run ragas-demo score
```

This is a paid RAGAS metric step. It writes the resumable checkpoint:

```text
.ragas-demo/checkpoints/scores.jsonl
```

Metrics produced per case:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

If interrupted:

```powershell
uv run ragas-demo score --resume
```

Before constructing the RAGAS scorer, the command verifies that `results/responses.jsonl` contains
exactly the current test-set cases. Resume also validates score checkpoints against the current
responses by `case_id`, prompt, reference, response, retrieved contexts, and source IDs.

## 8. Create Final Report Artifacts

Generate the final files:

```powershell
uv run ragas-demo report
```

Outputs:

```text
results/scores.csv
results/scores.json
results/report.md
```

The report command refuses to finalize if responses or scores are missing, duplicated, or mismatched
against the current test set.

## 9. Recovery Guide

Missing API key:

```powershell
Copy-Item .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

Missing or stale index:

```powershell
uv run ragas-demo ingest --rebuild
```

Interrupted answer collection:

```powershell
uv run ragas-demo collect-answers --resume
```

Interrupted scoring:

```powershell
uv run ragas-demo score --resume
```

Stale checkpoint after regenerating test cases:

```powershell
Remove-Item results\responses.jsonl
Remove-Item .ragas-demo\checkpoints\scores.jsonl
```

Only run those removals when you intentionally want to discard old paid-stage outputs.

## 10. Paid Call Boundary

Offline or local-only:

- `uv run pytest`
- `uv run ruff check app.py src tests`
- `uv run ragas-demo ingest`
- `uv run streamlit run app.py` startup only
- `uv run ragas-demo report`

Paid OpenAI calls:

- Asking a question in the Streamlit app
- `uv run ragas-demo smoke`
- `uv run ragas-demo generate-testset`
- `uv run ragas-demo collect-answers`
- `uv run ragas-demo score`
