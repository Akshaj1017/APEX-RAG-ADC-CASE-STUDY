"""Run the golden Q&A set against the API and report pass/fail.

This is the regression gate: `python eval/run_eval.py` ingests the sample
policies, asks every golden question, checks the expectations, and exits
non-zero if anything fails -- so it can block a CI pipeline.

Works offline with the fake provider or against the real model when
MISTRAL_API_KEY is set; the checks are identical either way.
"""
import glob
import os
import sys

# Allow running as a plain script (python eval/run_eval.py) from anywhere by
# putting the project root on the import path before importing the app package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from fastapi.testclient import TestClient
from app.db import init_db
from app.main import app
from app.services import ingestion

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_set.yaml")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _ensure_corpus() -> None:
    """Idempotently ingest the sample policies so the eval is self-contained."""
    for path in sorted(glob.glob(os.path.join(DATA, "*.pdf"))):
        with open(path, "rb") as fh:
            ingestion.ingest_pdf("eval", os.path.basename(path), fh.read())


def _check(case: dict, resp: dict) -> list[str]:
    """Return a list of failure reasons (empty == pass)."""
    failures: list[str] = []
    answer = resp.get("answer", "").lower()
    cited = " ".join((c.get("filename") or "").lower() for c in resp.get("citations", []))

    if case.get("expect_refusal"):
        if resp.get("grounded", True):
            failures.append("expected a refusal but the bot answered")
        return failures

    for needle in case.get("answer_contains", []):
        if needle.lower() not in answer:
            failures.append(f"answer missing '{needle}'")
    for needle in case.get("answer_excludes", []):
        if needle.lower() in answer:
            failures.append(f"answer wrongly contains '{needle}'")
    if case.get("must_cite") and case["must_cite"].lower() not in cited:
        failures.append(f"missing citation to '{case['must_cite']}'")
    return failures


def main() -> int:
    init_db()
    _ensure_corpus()
    cases = yaml.safe_load(open(GOLDEN))
    client = TestClient(app)

    passed = 0
    print(f"Running {len(cases)} golden cases\n" + "-" * 60)
    for case in cases:
        resp = client.post(
            "/chat", json={"conversation_id": "eval", "message": case["question"]}
        ).json()
        failures = _check(case, resp)
        if failures:
            print(f"FAIL  {case['id']}")
            for reason in failures:
                print(f"        - {reason}")
        else:
            passed += 1
            print(f"PASS  {case['id']}")

    print("-" * 60)
    print(f"{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
