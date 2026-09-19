"""
Value-shape collisions (GB-2d) — find same-slot facts the exact-key conflict
check cannot see.

``find_conflicts`` keys on (subject, relation), and GB-2c only loosens the
relation by name similarity. The live failure (EXP-QA-PATH idx=3) was one
subject holding three loan amounts under unrelated relation names — no shared
key, so nothing was ever superseded and the answerer had no freshness signal.

Detection here is vocabulary-free: classify each object into a shape class
(money, percent, duration, date, clock time, count) and pair same-subject,
same-shape, different-value facts. A Jev Score decides whether a pair states
the SAME attribute with different values; which one is newer is decided in
code from provenance dates (Jev reads dates as text, not as ordered values).
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

_NUM = r"\d[\d,]*(?:\.\d+)?"
SHAPES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("money", re.compile(rf"[$€£]\s?{_NUM}|\b{_NUM}\s?(?:dollars?|usd|euros?|pounds?|bucks)\b|\b{_NUM}\s?[km]\b", re.I)),
    ("percent", re.compile(rf"{_NUM}\s?(?:%|percent)", re.I)),
    ("clock", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")),
    ("frequency", re.compile(r"\b(?:once|twice|thrice|\w+\s+times)\s+(?:a|per|every)\s+\w+", re.I)),
    ("duration", re.compile(rf"\b(?:{_NUM}|an?|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]?"
                            r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b", re.I)),
    ("date", re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})\b", re.I)),
    ("count", re.compile(rf"^\s*(?:{_NUM}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)),
]

SAME_SLOT_LEVELS = [
    "They describe different things: a different attribute, item, event or person.",
    "Unclear whether they describe the same attribute of the same thing.",
    "They state the SAME attribute of the same thing with different values, so one is an update or "
    "correction of the other.",
]


def value_collisions_on() -> bool:
    return os.getenv("QA_VALUE_COLLISION", "").lower() == "jev"


def shape_of(text: str) -> Optional[str]:
    for name, pattern in SHAPES:
        if pattern.search(text or ""):
            return name
    return None


def _triple_date(triple: Any) -> str:
    meta = getattr(triple, "metadata", None) or {}
    if meta.get("session_date"):
        return str(meta["session_date"])
    refs = (meta.get("provenance") or {}).get("refs") or []
    dates = [str(r.get("document_date")) for r in refs if isinstance(r, dict) and r.get("document_date")]
    return max(dates) if dates else ""


def candidate_pairs(triples: List[Any], object_text: Callable[[Any], str], max_pairs: int = 40) -> List[Tuple[int, int]]:
    """Index pairs of same-subject, same-shape facts whose object values differ."""
    shaped: Dict[Tuple[str, str], List[int]] = {}
    for i, t in enumerate(triples):
        meta = getattr(t, "metadata", None) or {}
        if meta.get("superseded_by") or meta.get("superseded_at"):
            continue
        shape = shape_of(object_text(t))
        if shape:
            shaped.setdefault((t.subject, shape), []).append(i)
    pairs: List[Tuple[int, int]] = []
    for members in shaped.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = triples[members[x]], triples[members[y]]
                if object_text(a).strip().lower() != object_text(b).strip().lower():
                    pairs.append((members[x], members[y]))
    return pairs[:max_pairs]


def mark_stale_values(
    triples: List[Any],
    object_text: Callable[[Any], str],
    render: Callable[[Any], str],
    jev_client: Any,
    min_p_same: float = 0.5,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Return (triples with older same-slot values marked superseded, decisions).

    Marked triples are COPIES — the knowledge graph itself is never mutated here.
    """
    pairs = candidate_pairs(triples, object_text)
    if not pairs:
        return triples, []
    answers = jev_client.ask_many([
        ({"fact_a": render(triples[i]), "fact_b": render(triples[j])},
         {"same": {"type": "score", "instructions": "How do these two facts about the same subject relate?",
                   "criteria": SAME_SLOT_LEVELS}})
        for i, j in pairs])
    out = list(triples)
    decisions: List[Dict[str, Any]] = []
    for (i, j), ans in zip(pairs, answers):
        p_same = float(ans["same"]["probabilities"].get("2", 0.0))
        date_i, date_j = _triple_date(triples[i]), _triple_date(triples[j])
        decision = {"a": render(triples[i]), "b": render(triples[j]), "p_same": round(p_same, 3), "stale": None}
        if p_same >= min_p_same and date_i and date_j and date_i != date_j:
            stale, fresh = (i, j) if date_i < date_j else (j, i)
            meta = dict(getattr(out[stale], "metadata", None) or {})
            meta["superseded_by"] = f"{triples[fresh].subject}|{triples[fresh].relation}|{triples[fresh].object}"
            meta["superseded_reason"] = "query-time value collision (GB-2d)"
            out[stale] = dataclasses.replace(out[stale], metadata=meta)
            decision["stale"] = "a" if stale == i else "b"
        decisions.append(decision)
    return out, decisions
