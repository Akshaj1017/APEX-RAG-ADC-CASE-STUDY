"""Extract policy metadata from document text.

This is the single most important non-obvious piece of the prototype. Pure
semantic search retrieves BOTH the 2022 and 2025 travel policies for an "Asia
hotel limit" question, because they're equally *about* that topic. Similarity
cannot tell which one is current. Metadata can.

We capture four fields:
  policy_id      e.g. FIN-T&E-001   -> links different versions of the same policy
  version        e.g. 3.2           -> lets us order versions
  effective_date e.g. 2025-01-01    -> lets us prefer the most recent
  status         active | interim   -> self-declared lifecycle hint

`status` is only the document's *self-declared* hint. The authoritative
"superseded" flag is set later, at ingestion time, by comparing versions that
share a policy_id -- see ingestion.py.
"""
import re

_ID = re.compile(r"(?:Policy\s*ID|Reference)\s*:\s*([A-Z0-9][A-Z0-9&\-]+)", re.I)
_VERSION = re.compile(r"Version\s*:\s*([0-9]+(?:\.[0-9]+)*)", re.I)
_EFFECTIVE = re.compile(r"Effective\s*Date\s*:\s*(.+)", re.I)
_DATE_FALLBACK = re.compile(r"\bDate\s*:\s*(.+)", re.I)

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

_INTERIM_HINTS = ("interim", "temporary measures", "historical note",
                   "pending the release", "covid")


def extract_metadata(text: str, filename: str) -> dict[str, str | None]:
    head = text[:1500]                       # metadata always lives in the first lines

    policy_id = _first(_ID, head)
    version = _first(_VERSION, head)

    raw_date = _first(_EFFECTIVE, head) or _first(_DATE_FALLBACK, head)
    effective_date = _normalize_date(raw_date) if raw_date else None

    low = text.lower()
    status = "interim" if any(h in low for h in _INTERIM_HINTS) else "active"
    if "supersede" in low or "replaces earlier" in low or "replaces prior" in low:
        status = "active"

    return {"policy_id": policy_id, "version": version,
            "effective_date": effective_date, "status": status}


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _normalize_date(raw: str) -> str | None:
    """'1 January 2025' -> '2025-01-01'; 'July 2019' -> '2019-07-01'."""
    raw = raw.strip()
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        day, mon, year = int(m.group(1)), m.group(2).lower(), m.group(3)
        if mon in _MONTHS:
            return f"{year}-{_MONTHS[mon]:02d}-{day:02d}"
    m = re.search(r"([A-Za-z]+)\s+(\d{4})", raw)
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(2)}-{_MONTHS[m.group(1).lower()]:02d}-01"
    return None
