"""
Question-intent routing (GB-16) — one Jev Choice per question.

The focused top-k retrieval is right for point lookups and structurally wrong
for aggregation questions ("how many…", "total…", "list all…"), which need
every matching fact (DIAG-LME-FAILURES RC1). The intent decides how evidence is
selected and whether the abstain gate may fire:

    point / boolean   top-k evidence, gate allowed
    aggregate         every relevant fact (no top-k cap), gate exempt
    temporal          top-k evidence ordered by date, gate allowed
    inferential       top-k evidence, gate exempt (answers are inferred, so no
                      single fact states them)

Enabled with ``QA_INTENT_ROUTER=jev``; any failure degrades to ``point``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

INTENTS = {
    "point": "asks for one specific fact: a name, place, thing, value or short description",
    "aggregate": "asks to count, total, sum, list all, or compare across several separate events or items "
                 "(how many, how much in total, which ones, all the)",
    "temporal": "asks when something happened, how long ago, how long it lasted, or the order of events",
    "inferential": "asks for a judgement, likely preference, recommendation or what someone would probably "
                   "do or feel — it must be inferred rather than looked up",
    "boolean": "a yes/no question about whether something is true",
}
GATE_EXEMPT = {"aggregate", "inferential"}
AGGREGATE_RIDER = (
    "AGGREGATION QUESTION: first enumerate EVERY distinct matching item or event found in the facts "
    "(one per line, with its date if shown), ignoring superseded facts and duplicates of the same event; "
    "then count / sum / compare them to give the final answer."
)


def intent_router_on() -> bool:
    return os.getenv("QA_INTENT_ROUTER", "").lower() == "jev"


def classify_intent(query: str, jev_client: Optional[Any] = None) -> str:
    """Return one of INTENTS; ``point`` when the router is off or Jev fails."""
    if not intent_router_on():
        return "point"
    try:
        if jev_client is None:
            from multi_agent_kg.core.retrievers import _jev_client

            jev_client = _jev_client()
        answer = jev_client.ask(
            {"question": query},
            {"intent": {"type": "choice", "instructions": "What kind of question is this?", "criteria": INTENTS}},
        )["intent"]
        return answer["choice"] if answer["choice"] in INTENTS else "point"
    except Exception as exc:
        print(f"  WARNING: intent routing failed ({exc}); treating as point lookup")
        return "point"
