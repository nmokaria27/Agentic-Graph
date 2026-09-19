#!/usr/bin/env python3
"""EXP-JEV-2 / T3 — reranker + Jev cascade retrieval with an abstain gate (LoCoMo).

Same cached KG, answer model and prompt for every arm; facts carry their
source sentence. Only evidence selection differs:

B0  production hybrid retriever, top ``--k``
R1  qwen3-reranker scores EVERY active triple (chunked, parallel), top ``--k``
R2  cascade: reranker top ``--pool`` → one Jev Noul per fact → top ``--k``
    (Jev sees 60 facts instead of the whole KG: ~15x less Jev spend than a scan)
D   no reranker: embedding top ``--dense-pool`` (vectors already built, ~free)
    → one Jev Noul per fact → top ``--k``
H   hybrid: R2 top ``k-5`` + production retriever top 5 (keeps lexical/date hits)

Gate signals are recorded per question so abstain thresholds can be swept
offline (``tune``) on one conversation and applied frozen (``--gates``) to another:
rr_max (free), jev_max, jev_direct (one extra Jev Noul on the selected set).

    python scripts/jev/fw_exec.py python scripts/jev/t3_locomo_retrieval.py run --sample conv-30
    python scripts/jev/t3_locomo_retrieval.py tune --rows evaluation/results/jev/t3_conv-30.json
    python scripts/jev/fw_exec.py python scripts/jev/t3_locomo_retrieval.py run --sample conv-26 \
        --gates evaluation/results/jev/t3_gates.json
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "jev"))

from t2_locomo_rerank import (  # noqa: E402
    ABSTAIN, CAT_NAMES, OUT_DIR, _answer, _fact, _gold, _pick_questions, _question_text, _token_recall, score_qa,
)

ARMS = ("B0", "R1", "R2", "H", "D")
SIGNALS = {"B0": ("rr_max",), "R1": ("rr_max", "jev_direct"), "R2": ("rr_max", "jev_max", "jev_direct"),
           "H": ("rr_max", "jev_max", "jev_direct"), "D": ("jev_max",)}
for _arm in list(SIGNALS):
    SIGNALS[_arm] = SIGNALS[_arm] + ("jev_answer_ok",)


def _aggregate(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[CAT_NAMES[row["category"]]].append(row[key]["score"])
    non_adv = [r[key]["score"] for r in rows if r["category"] != 5]
    return {
        "overall": round(sum(r[key]["score"] for r in rows) / len(rows), 3),
        "overall_ex_adversarial": round(sum(non_adv) / max(1, len(non_adv)), 3),
        "by_category": {c: round(sum(v) / len(v), 3) for c, v in by_cat.items()},
    }


def _gated_score(row: Dict[str, Any], arm: str, signal: str, tau: float) -> float:
    if row["signals"][arm][signal] < tau:
        return score_qa(ABSTAIN, row["gold"], row["category"])
    return row[arm]["score"]


def tune(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.rows).read_text())["rows"]
    gates = {}
    for arm, signals in SIGNALS.items():
        best = (sum(r[arm]["score"] for r in rows) / len(rows), None, 0.0)
        for signal in signals:
            for tau in sorted({round(r["signals"][arm][signal], 3) for r in rows}):
                mean = sum(_gated_score(r, arm, signal, tau) for r in rows) / len(rows)
                if mean > best[0] + 1e-9:
                    best = (mean, signal, tau)
        gates[arm] = {"signal": best[1], "tau": best[2], "tuned_overall": round(best[0], 3)}
    Path(args.output).write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))


def run(args: argparse.Namespace) -> None:
    from multi_agent_kg.core import LLMConfig
    from multi_agent_kg.core.config import RetrievalConfig
    from multi_agent_kg.core.kg_operations import load_governed_kg
    from multi_agent_kg.core.qa_orchestrator import QAOrchestrator
    from multi_agent_kg.core.vector_index import KGVectorStore
    from multi_agent_kg.llm.rerank_client import RerankClient
    from multi_agent_kg.llm.typesafe_client import JevClient

    sample = next(s for s in json.loads(Path(args.data_file).read_text()) if s["sample_id"] == args.sample)
    questions = _pick_questions(sample["qa"], args.per_category)
    gkg = load_governed_kg(str(Path(args.kg_dir) / args.sample / "governed_kg.json"))
    retrieval = RetrievalConfig()
    vec_dir = OUT_DIR / f"vectors_{args.sample}"
    store = KGVectorStore.load_dir(str(vec_dir), gkg, model=retrieval.embedding_model)
    if store is None:
        store = KGVectorStore(model=retrieval.embedding_model)
        store.build(gkg)
        store.save_dir(str(vec_dir))
    expert = QAOrchestrator(governed_kg=gkg, llm_config=LLMConfig(model=args.model), vector_store=store,
                            retrieval_config=retrieval).global_fallback_expert
    all_facts = [_fact(t, True) for t in gkg.kg.get_active_triples()]
    timing: Dict[str, float] = {}

    t0 = time.perf_counter()
    selections: Dict[str, List[List[str]]] = {
        "B0": [[_fact(t, True) for t in expert._query_focused_triples(qa["question"], limit=args.k)]
               for qa in questions]}
    timing["B0_retrieve_s"] = round(time.perf_counter() - t0, 2)

    # R1: reranker over the whole KG — questions in parallel, chunks in parallel inside each.
    reranker = RerankClient(cache_dir=str(OUT_DIR / "rerank_cache"))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rr_scores = list(pool.map(lambda qa: reranker.score(qa["question"], all_facts), questions))
    reranker.save_cache()
    timing["R1_rerank_s"] = round(time.perf_counter() - t0, 2)
    rr_orders = [sorted(range(len(all_facts)), key=lambda i: (-s[i], i)) for s in rr_scores]
    selections["R1"] = [[all_facts[i] for i in order[: args.k]] for order in rr_orders]

    # R2: Jev re-scores only the reranker's top pool.
    jev = JevClient(cache_dir=str(OUT_DIR / "jev_cache"), usage_log=str(OUT_DIR / "jev_usage.jsonl"),
                    max_questions_per_call=128)
    t0 = time.perf_counter()
    pools = [[all_facts[i] for i in order[: args.pool]] for order in rr_orders]
    jev_fact = jev.ask_many([
        ({"question": qa["question"]}, {
            f"f{i}": {"type": "noul",
                      "instructions": "Is this fact from the conversation useful evidence for answering the "
                                      f"question? Fact: {fact}"}
            for i, fact in enumerate(pool)})
        for qa, pool in zip(questions, pools)])
    jev_p = [[ans[f"f{i}"]["p"] for i in range(len(pool))] for pool, ans in zip(pools, jev_fact)]
    r2_orders = [sorted(range(len(p)), key=lambda i: (-p[i], i))[: args.k] for p in jev_p]
    selections["R2"] = [[pool[i] for i in order] for pool, order in zip(pools, r2_orders)]
    # D: dense (embedding) shortlist → Jev. One batched Jev request per question.
    active = gkg.kg.get_active_triples()
    d_pools = [[_fact(t, True) for t in expert._vector_seed_triples(qa["question"], active, top_k=args.dense_pool)]
               for qa in questions]
    d_ans = jev.ask_many([
        ({"question": qa["question"]}, {
            f"f{i}": {"type": "noul",
                      "instructions": "Is this fact from the conversation useful evidence for answering the "
                                      f"question? Fact: {fact}"}
            for i, fact in enumerate(pool)})
        for qa, pool in zip(questions, d_pools)])
    d_p = [[ans[f"f{i}"]["p"] for i in range(len(pool))] for pool, ans in zip(d_pools, d_ans)]
    selections["D"] = [[pool[i] for i in sorted(range(len(p)), key=lambda i: (-p[i], i))[: args.k]]
                       for pool, p in zip(d_pools, d_p)]
    selections["H"] = []
    for r2, b0 in zip(selections["R2"], selections["B0"]):
        keep = r2[: max(0, args.k - 5)]
        selections["H"] = selections["H"] + [keep + [f for f in b0 if f not in keep][:5]]
    direct = {arm: jev.ask_many([
        ({"question": qa["question"], "facts": facts},
         {"direct": {"type": "noul",
                     "instructions": "Do these facts explicitly state the specific answer to the question? "
                                     "Answer no if they are only about a related topic or a different person."}})
        for qa, facts in zip(questions, selections[arm])]) for arm in ("R1", "R2", "H")}
    timing["R2_jev_s"] = round(time.perf_counter() - t0, 2)

    predictions = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as pool:
            predictions[arm] = list(pool.map(
                lambda i: _answer(args.model, _question_text(questions[i]), selections[arm][i]),
                range(len(questions))))
        timing[f"{arm}_answer_s"] = round(time.perf_counter() - t0, 2)

    # Answer-verification gate: Jev checks the proposed answer against the facts.
    t0 = time.perf_counter()
    answer_ok = {}
    for arm in ARMS:
        res = jev.ask_many([
            ({"question": qa["question"], "facts": selections[arm][i], "proposed_answer": predictions[arm][i]},
             {"ok": {"type": "noul",
                     "instructions": "Do the facts explicitly support that the proposed answer correctly answers "
                                     "the question, about the same person and event the question asks about?"}})
            for i, qa in enumerate(questions)])
        answer_ok[arm] = [r["ok"]["p"] for r in res]
    timing["answer_gate_s"] = round(time.perf_counter() - t0, 2)

    gates = json.loads(Path(args.gates).read_text()) if args.gates else {}
    rows = []
    for i, qa in enumerate(questions):
        row: Dict[str, Any] = {"question": qa["question"], "gold": _gold(qa), "category": qa["category"],
                               "signals": {
                                   "B0": {"rr_max": rr_scores[i][rr_orders[i][0]]},
                                   "R1": {"rr_max": rr_scores[i][rr_orders[i][0]],
                                          "jev_direct": direct["R1"][i]["direct"]["p"]},
                                   "R2": {"rr_max": rr_scores[i][rr_orders[i][0]],
                                          "jev_max": max(jev_p[i]) if jev_p[i] else 0.0,
                                          "jev_direct": direct["R2"][i]["direct"]["p"]},
                                   "H": {"rr_max": rr_scores[i][rr_orders[i][0]],
                                         "jev_max": max(jev_p[i]) if jev_p[i] else 0.0,
                                         "jev_direct": direct["H"][i]["direct"]["p"]}}}
        row["signals"]["D"] = {"jev_max": max(d_p[i]) if d_p[i] else 0.0}
        for arm in ARMS:
            row["signals"][arm]["jev_answer_ok"] = answer_ok[arm][i]
            row[arm] = {"pred": predictions[arm][i],
                        "score": score_qa(predictions[arm][i], _gold(qa), qa["category"]),
                        "evidence_recall": _token_recall(_gold(qa), " ".join(selections[arm][i]))}
        for arm, gate in gates.items():
            if gate.get("signal"):
                row[f"{arm}+gate"] = {"score": _gated_score(row, arm, gate["signal"], gate["tau"])}
        rows.append(row)

    arms = {key: _aggregate(rows, key) for key in rows[0] if key in ARMS or key.endswith("+gate")}
    result = {"config": vars(args), "n_questions": len(questions), "kg_triples": len(all_facts), "timing": timing,
              "jev": {**jev.stats, "cost_usd": jev.cost_usd()}, "reranker": reranker.stats,
              "gates": gates, "arms": arms, "rows": rows}
    (OUT_DIR / f"t3_{args.sample}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kg-dir", default="../Agent-Graph-Memory/evaluation/results/locomo_kg_cache")
    r.add_argument("--data-file", default="evaluation/LoComo/data/locomo10.json")
    r.add_argument("--sample", default="conv-26")
    r.add_argument("--per-category", type=int, default=8)
    r.add_argument("--pool", type=int, default=60)
    r.add_argument("--k", type=int, default=15)
    r.add_argument("--dense-pool", type=int, default=200)
    r.add_argument("--model", default=os.getenv("LLM_DEFAULT_MODEL"))
    r.add_argument("--gates", help="frozen gates JSON from `tune`")
    t = sub.add_parser("tune")
    t.add_argument("--rows", required=True)
    t.add_argument("--output", default=str(OUT_DIR / "t3_gates.json"))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(args) if args.mode == "run" else tune(args)


if __name__ == "__main__":
    main()
