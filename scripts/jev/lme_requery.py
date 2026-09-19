#!/usr/bin/env python3
"""EXP-JEV-3 — re-ask cached LongMemEval questions under a named QA config.

Never writes into the source cache: the first run copies ``--src-dir`` to a NEW
``--dst-dir`` (refuses to reuse an existing one unless ``--resume``), then
re-asks every question there with ``--model`` and the config's env flags, in
parallel. Score afterwards with evaluation/LongMemEval/score_longmemeval.py.

    python scripts/jev/lme_requery.py --config base --ability knowledge-update \
        --model accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation", "LongMemEval"))

CONFIGS = {
    "base": {"RETRIEVAL_MODE": "hybrid"},
    "evidence": {"RETRIEVAL_MODE": "hybrid", "QA_FACT_EVIDENCE": "1"},
    "cascade": {"RETRIEVAL_MODE": "jev_cascade", "QA_FACT_EVIDENCE": "1"},
    "cascade_router": {"RETRIEVAL_MODE": "jev_cascade", "QA_FACT_EVIDENCE": "1", "QA_INTENT_ROUTER": "jev"},
    "cascade_full": {"RETRIEVAL_MODE": "jev_cascade", "QA_FACT_EVIDENCE": "1", "QA_INTENT_ROUTER": "jev",
                     "QA_VALUE_COLLISION": "jev"},
    "cascade_gate": {"RETRIEVAL_MODE": "jev_cascade", "QA_FACT_EVIDENCE": "1", "QA_ABSTAIN_GATE": "jev"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--ability", required=True, help="e.g. knowledge-update")
    ap.add_argument("--src-root", default=os.path.expanduser("~/Agentic-Graph-Memory/evaluation/results"))
    ap.add_argument("--dst-root", default="evaluation/results/jev_lme2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ.update(CONFIGS[args.config])

    src = os.path.join(args.src_root, f"lme_breadth_{args.ability}")
    dst = os.path.join(args.dst_root, f"{args.ability}__{args.config}")
    if os.path.exists(dst) and not args.resume:
        sys.exit(f"{dst} exists; pass --resume to continue it (nothing is overwritten or deleted)")
    if not os.path.exists(dst):
        os.makedirs(args.dst_root, exist_ok=True)
        shutil.copytree(src, dst)

    from requery_cached import best_kg_snapshot, load_wrapper_class

    from multi_agent_kg.core import GovernedKnowledgeGraph

    Wrapper = load_wrapper_class()

    # Every config must answer over the SAME domain map and embeddings, and
    # rebuilding them per config is what dominates wall time — so both are built
    # once per question under <dst-root>/_shared and reused by every config.
    import threading

    from multi_agent_kg.core.governance import OrgChart
    from multi_agent_kg.core.vector_index import KGVectorStore

    adapter_globals = Wrapper._build_qa_system.__globals__
    RealBuilder = adapter_globals["DomainBuilder"]
    local = threading.local()

    class CachedDomainBuilder:
        def __init__(self, llm_config):
            self._real = RealBuilder(llm_config)

        def build(self, kg, *a, **k):
            path = os.path.join(local.shared, "org_chart.json")
            if os.path.exists(path):
                return OrgChart.from_dict(json.load(open(path)), kg)
            org = self._real.build(kg, *a, **k)
            with open(path, "w") as f:
                json.dump(org.to_dict(), f)
            return org

    adapter_globals["DomainBuilder"] = CachedDomainBuilder

    def shared_vector_store(gkg, model):
        vec_dir = os.path.join(local.shared, "vectors")
        store = KGVectorStore.load_dir(vec_dir, gkg, model=model)
        if store is None:
            store = KGVectorStore(model=model)
            store.build(gkg)
            store.save_dir(vec_dir)
        return store

    def one(cache_path: str) -> str:
        record = json.load(open(cache_path))
        if record.get("requery", {}).get("config") == args.config:
            return f"{record['question_id']}: already done"
        idx = record["idx"]
        snapshot, _ = best_kg_snapshot(os.path.join(dst, "ckpt", f"context_{idx}"))
        if snapshot is None:
            return f"{record['question_id']}: NO SNAPSHOT"
        gkg = GovernedKnowledgeGraph.from_dict(snapshot)
        local.shared = os.path.join(args.dst_root, "_shared", args.ability, record["question_id"])
        os.makedirs(local.shared, exist_ok=True)
        # The adapter ignores RETRIEVAL_MODE from the environment: pass it explicitly.
        wrapper = Wrapper(model=args.model, answer_format="longmemeval",
                          retrieval_mode=CONFIGS[args.config]["RETRIEVAL_MODE"],
                          extraction_mode=record.get("extraction_mode", "wide"))
        gkg.vector_store = shared_vector_store(gkg, wrapper.retrieval_config.embedding_model)
        wrapper._context_id = idx
        wrapper._governed_kg = gkg
        wrapper._qa_system = wrapper._build_qa_system(gkg)
        wrapper._qa_system.governed_kg = gkg
        t0 = time.time()
        resp = wrapper.send_message(f"The current date is {record['question_date']}. {record['question']}",
                                    memorizing=False, query_id=0, context_id=idx)
        record["hypothesis_pre_requery"] = record.get("hypothesis", "")
        record["hypothesis"] = resp.get("output", "")
        record["model"] = args.model
        record["requery"] = {"config": args.config, "env": CONFIGS[args.config],
                             "mode_used": wrapper.retrieval_config.retrieval_mode,
                             "query_s": round(time.time() - t0, 1)}
        with open(cache_path, "w") as f:
            json.dump(record, f, indent=2)
        return f"{record['question_id']}: {record['requery']['query_s']}s hyp={record['hypothesis'][:90]!r} gold={record['answer'][:50]!r}"

    paths = sorted(p for p in glob.glob(os.path.join(dst, "*.json")))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(one, paths):
            print("[LME]", line, flush=True)
    print("LME_REQUERY_DONE", dst)


if __name__ == "__main__":
    main()
