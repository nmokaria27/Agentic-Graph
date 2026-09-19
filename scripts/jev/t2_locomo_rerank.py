#!/usr/bin/env python3
"""EXP-JEV-1 / T2 — Jev on the retrieval side (LoCoMo, cached KG, requery only).

Controlled A/B: every arm uses the SAME candidate pool (the production
hybrid retriever's top-``--pool`` focused triples from
``QAOrchestrator.global_fallback_expert._query_focused_triples``), the SAME
small answer model and the SAME prompt. Only evidence selection differs:

B0  baseline   first ``--k`` triples in the retriever's own order
B1  jev_rerank ``--pool`` candidates re-ordered by one Jev Noul per triple
               ("is this fact useful to answer the question?"), top ``--k``
B2  jev_gate   B1 + one Jev Noul over the selected evidence ("do these facts
               contain the answer?"); below ``--abstain-p`` → abstain
B3  jev_scan   no retriever pool: Jev scores EVERY active triple in the KG for
               the question (Jev is cheap enough to full-scan a small KG), top ``--k``

Also reports an LLM-free retrieval metric: gold-answer token recall inside
the selected evidence.

    python scripts/jev/fw_exec.py python scripts/jev/t2_locomo_rerank.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation", "LoComo"))

from run_eval import _normalize as normalize_answer, score_qa  # noqa: E402

from multi_agent_kg.core import LLMConfig  # noqa: E402
from multi_agent_kg.core.config import RetrievalConfig  # noqa: E402
from multi_agent_kg.core.kg_operations import load_governed_kg  # noqa: E402
from multi_agent_kg.core.qa_orchestrator import QAOrchestrator, _format_triple  # noqa: E402
from multi_agent_kg.core.vector_index import KGVectorStore  # noqa: E402
from multi_agent_kg.llm.openai_client import chat_completion  # noqa: E402
from multi_agent_kg.llm.typesafe_client import JevClient  # noqa: E402

OUT_DIR = Path("evaluation/results/jev")
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ABSTAIN = "No information available"

ANSWER_PROMPT = """You answer questions about a long conversation between two people, using ONLY the
knowledge-graph facts below (extracted from the conversation). Reply with the shortest possible
answer span (a name, date, place, short phrase or comma-separated list) — no explanation.
If the facts do not contain the answer, reply exactly: {abstain}

Facts:
{facts}

Question: {question}
Answer:"""


def _question_text(qa: Dict[str, Any]) -> str:
    q = qa["question"]
    if qa["category"] == 2:
        q += " Use DATE OF CONVERSATION to answer with an approximate date."
    return q


def _gold(qa: Dict[str, Any]) -> str:
    return str(qa.get("answer", qa.get("adversarial_answer", "")))


def _token_recall(gold: str, evidence: str) -> float:
    gold_toks = normalize_answer(gold).split()
    if not gold_toks:
        return 0.0
    ev = set(normalize_answer(evidence).split())
    return sum(t in ev for t in gold_toks) / len(gold_toks)


def _answer(model: str, question: str, facts: List[str]) -> str:
    prompt = ANSWER_PROMPT.format(abstain=ABSTAIN, facts="\n".join(f"- {f}" for f in facts) or "(none)",
                                  question=question)
    out = chat_completion(messages=[{"role": "user", "content": prompt}], model=model,
                          temperature=0.0, max_tokens=1024)
    lines = [line.strip() for line in str(out or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _fact(triple: Any, with_evidence: bool) -> str:
    text = _format_triple(triple)
    evidence = ((getattr(triple, "metadata", None) or {}).get("evidence") or "").strip()
    return f"{text} — \"{evidence}\"" if with_evidence and evidence else text


def _pick_questions(qa_list: List[Dict[str, Any]], per_cat: int) -> List[Dict[str, Any]]:
    by_cat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for qa in qa_list:
        if len(by_cat[qa["category"]]) < per_cat:
            by_cat[qa["category"]].append(qa)
    return [qa for cat in sorted(by_cat) for qa in by_cat[cat]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kg-dir", default="../Agent-Graph-Memory/evaluation/results/locomo_kg_cache")
    parser.add_argument("--data-file", default="evaluation/LoComo/data/locomo10.json")
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--pool", type=int, default=60)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--abstain-p", type=float, default=0.1)
    parser.add_argument("--model", default=os.getenv("LLM_DEFAULT_MODEL"))
    parser.add_argument("--with-evidence", action="store_true",
                        help="render each fact with its source sentence (all arms)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = next(s for s in json.loads(Path(args.data_file).read_text()) if s["sample_id"] == args.sample)
    questions = _pick_questions(sample["qa"], args.per_category)
    gkg = load_governed_kg(str(Path(args.kg_dir) / args.sample / "governed_kg.json"))

    # Embed once, reuse across runs.
    retrieval = RetrievalConfig()
    vec_dir = OUT_DIR / f"vectors_{args.sample}"
    store = KGVectorStore.load_dir(str(vec_dir), gkg, model=retrieval.embedding_model)
    if store is None:
        store = KGVectorStore(model=retrieval.embedding_model)
        store.build(gkg)
        store.save_dir(str(vec_dir))
    orch = QAOrchestrator(governed_kg=gkg, llm_config=LLMConfig(model=args.model),
                          vector_store=store, retrieval_config=retrieval)
    expert = orch.global_fallback_expert

    started = time.perf_counter()
    pools = [expert._query_focused_triples(qa["question"], limit=args.pool) for qa in questions]
    retrieval_s = time.perf_counter() - started

    jev = JevClient(cache_dir=str(OUT_DIR / "jev_cache"), usage_log=str(OUT_DIR / "jev_usage.jsonl"),
                    max_questions_per_call=128)
    started = time.perf_counter()
    fact_lists = [[_fact(t, args.with_evidence) for t in pool] for pool in pools]
    rerank_answers = jev.ask_many([
        ({"question": qa["question"]}, {
            f"f{i}": {"type": "noul",
                      "instructions": "Is this fact from the conversation useful evidence for answering the "
                                      f"question? Fact: {fact}"}
            for i, fact in enumerate(facts)})
        for qa, facts in zip(questions, fact_lists)
    ])
    all_facts = [_fact(t, args.with_evidence) for t in gkg.kg.get_active_triples()]
    scan_answers = jev.ask_many([
        ({"question": qa["question"]}, {
            f"f{i}": {"type": "noul",
                      "instructions": "Is this fact from the conversation useful evidence for answering the "
                                      f"question? Fact: {fact}"}
            for i, fact in enumerate(all_facts)})
        for qa in questions
    ])
    selections: Dict[str, List[List[str]]] = {"B0": [], "B1": [], "B3": []}
    for answers in scan_answers:
        order = sorted(range(len(all_facts)), key=lambda i: (-answers[f"f{i}"]["p"], i))
        selections["B3"].append([all_facts[i] for i in order[: args.k]])
    for facts, answers in zip(fact_lists, rerank_answers):
        selections["B0"].append(facts[: args.k])
        order = sorted(range(len(facts)), key=lambda i: (-answers[f"f{i}"]["p"], i))
        selections["B1"].append([facts[i] for i in order[: args.k]])
    gate_answers = jev.ask_many([
        ({"question": qa["question"], "facts": facts},
         {"answerable": {"type": "noul",
                         "instructions": "Do these facts contain the information needed to answer the question?"}})
        for qa, facts in zip(questions, selections["B1"])
    ])
    jev_s = time.perf_counter() - started

    def run_arm(arm: str) -> List[str]:
        facts_list = selections["B1" if arm == "B2" else arm]

        def one(i: int) -> str:
            if arm == "B2" and gate_answers[i]["answerable"]["p"] < args.abstain_p:
                return ABSTAIN
            return _answer(args.model, _question_text(questions[i]), facts_list[i])

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(one, range(len(questions))))

    arms: Dict[str, Any] = {}
    predictions = {}
    for arm in ("B0", "B1", "B2", "B3"):
        t0 = time.perf_counter()
        predictions[arm] = run_arm(arm)
        arms[arm] = {"answer_wall_s": round(time.perf_counter() - t0, 2)}

    rows = []
    for i, qa in enumerate(questions):
        row = {"question": qa["question"], "gold": _gold(qa), "category": qa["category"],
               "p_answerable": gate_answers[i]["answerable"]["p"],
               "pool_recall": _token_recall(_gold(qa), " ".join(fact_lists[i])),
               "kg_recall": _token_recall(_gold(qa), " ".join(all_facts))}
        for arm in ("B0", "B1", "B2", "B3"):
            facts = selections["B1" if arm == "B2" else arm][i]
            row[arm] = {"pred": predictions[arm][i],
                        "score": score_qa(predictions[arm][i], _gold(qa), qa["category"]),
                        "evidence_recall": _token_recall(_gold(qa), " ".join(facts))}
        rows.append(row)

    ceilings = {}
    for key in ("pool_recall", "kg_recall"):
        per = defaultdict(list)
        for row in rows:
            per[CAT_NAMES[row["category"]]].append(row[key])
        ceilings[key] = {c: round(sum(v) / len(v), 3) for c, v in per.items()}
    for arm in ("B0", "B1", "B2", "B3"):
        by_cat, ev_by_cat = defaultdict(list), defaultdict(list)
        for row in rows:
            by_cat[CAT_NAMES[row["category"]]].append(row[arm]["score"])
            ev_by_cat[CAT_NAMES[row["category"]]].append(row[arm]["evidence_recall"])
        non_adv = [r[arm]["score"] for r in rows if r["category"] != 5]
        arms[arm].update({
            "overall": round(sum(r[arm]["score"] for r in rows) / len(rows), 3),
            "overall_ex_adversarial": round(sum(non_adv) / max(1, len(non_adv)), 3),
            "by_category": {c: round(sum(v) / len(v), 3) for c, v in by_cat.items()},
            "evidence_recall_by_category": {c: round(sum(v) / len(v), 3) for c, v in ev_by_cat.items()},
        })
    result = {
        "config": vars(args), "n_questions": len(questions),
        "retrieval_s": round(retrieval_s, 2),
        "jev": {**jev.stats, "cost_usd": jev.cost_usd(), "wall_s": round(jev_s, 2)},
        "evidence_ceilings": ceilings, "arms": arms, "rows": rows,
    }
    (OUT_DIR / f"t2_{args.sample}{'_ev' if args.with_evidence else ''}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
