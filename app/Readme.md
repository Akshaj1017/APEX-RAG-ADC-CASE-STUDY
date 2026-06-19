# Apex RAG — Internal Policy Assistant

A small RAG API that answers questions about company policy PDFs and shows which document the answer came from.

Stack: Python 3.12, FastAPI, SQLite, ChromaDB, Mistral API.

The interesting part: some policies in the set contradict each other. For example the travel policy `FIN-T&E-001` has two versions — the 2022 one says the Asia lodging limit is USD 200, the 2025 one says USD 180. A basic RAG system would find both and might quote the old number. This one reads each document's version and date when it's uploaded, figures out which version is current, and only answers from the up-to-date one.

## What the API does

- `POST /upload` — send a PDF, it gets parsed, split, embedded, and indexed.
- `POST /chat` — ask a question, get an answer with a citation.
- `GET /health` — simple "is it running" check.
- `GET /docs` — interactive page to try the API in the browser.

## How to run it (Windows / PowerShell)

From inside the `apex-rag` folder:

```powershell
# 1. create and activate the virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get *"running scripts is disabled on this system"*, run this once and try the activate line again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

When it works, your prompt starts with `(.venv)`.

```powershell
# 2. (optional) add a Mistral key — it also runs fine without one
copy .env.example .env        # then set MISTRAL_API_KEY=... inside .env

# 3. install dependencies
pip install -r requirements.txt

# 4. start the server
uvicorn app.main:app --reload
```

Now open **http://localhost:8000/docs** in your browser. That's the page where you can use the API.

To load all 8 policy PDFs at once, open a **second terminal** (with `.venv` active too) and run:

```powershell
Get-ChildItem data\*.pdf | ForEach-Object {
    curl.exe -s -X POST localhost:8000/upload -F "conversation_id=demo" -F "file=@$($_.FullName)"
}
```

Then go to `/docs`, open `POST /chat`, and ask something. For example:
*"What is the travel reimbursement limit for Asia?"* → answer is **USD 180**, citing `reimbursement_2025.pdf` (the 2025 version), not the old USD 200.

## Running the tests

The eval runs in-process (no server needed). It sends a fixed set of questions from `eval/golden_set.yaml` through the app, checks the answers, and prints a pass/fail line per question with a final score. The key checks are the contradiction cases — it confirms the answers use the current policy (e.g. Asia lodging is `180`, not the old `200`) and fails if a retired value ever shows up.

Make sure your Mistral key is set in `.env`, then run:

```powershell
.venv\Scripts\Activate.ps1
python eval\run_eval.py
```

If the run stalls partway, that's the Mistral free-tier rate limit — press `Ctrl + C` and run it again.

## Design Decisions

The full reasoning is in `apex-rag-design-doc.md`. The short version:

**Handling contradictory documents.** This is the main idea. Search by meaning alone can't tell which version of a policy is current — two versions both look relevant. So when a PDF is uploaded, we pull out its policy ID, version, date, and status. If a newer version of the same policy already exists, the older one gets marked as outdated and is left out of answers. Older "interim" memos are kept but ranked lower. We use simple pattern matching to read these fields, which is fast and predictable for these document headers. An AI model could read messier documents better, but it would make every upload slower and more expensive.

**Two kinds of search combined.** Meaning-based search is good at "Asia hotel cap" → "lodging reimbursement limit." Keyword search is good at exact things like a policy ID or a dollar amount. We run both and merge the results so we don't miss either kind of match.

**ChromaDB instead of FAISS.** FAISS is faster but only stores vectors — no metadata, no saving to disk. Since the whole point here is using metadata (version, date) to pick the right document, Chroma fits better. For a real 10,000-person rollout we'd move to a managed database.

**SQLite for the rest.** No setup, no separate server to run. Easy to swap for Postgres later by changing one line.

**Guardrails against made-up answers.** Before calling the model, a quick check sees if the question actually overlaps with the documents found — if not, it says it doesn't know instead of guessing. The model is also told to answer only from the provided text.

**Runs without an API key.** Every external piece (embeddings, the chat model, the vector store) has a local fallback, so the whole thing works offline. Good for testing; for a real demo you'd use the Mistral key.

**Left out on purpose:** streaming replies, follow-up question rewriting, reranking, a chat web page, Docker, and scanned-PDF support. For a prototype, the value is in answering correctly and handling the contradictions — these extras can be added later.