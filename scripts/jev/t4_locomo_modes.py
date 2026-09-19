#!/usr/bin/env python3
"""EXP-JEV-3 quick check — the REAL QA path (QAOrchestrator.query) under two configs.

    base : RETRIEVAL_MODE=hybrid (production)
    jev  : RETRIEVAL_MODE=jev_cascade + QA_FACT_EVIDENCE=1 + QA_ABSTAIN_GATE=jev

Same cached LoCoMo KG, same answer model, LoCoMo scorer. Run once per config:

    python scripts/jev/fw_exec.py python scripts/jev/t4_locomo_modes.py --config base
    python scripts/jev/fw_exec.py python scripts/jev/t4_locomo_modes.py --config jev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "jev"))

CONFIGS = {
    "base": {"RETRIEVAL_MODE": "hybrid"},
    "jev": {"RETRIEVAL_MODE": "jev_cascade", "QA_FACT_EVIDENCE": "1", "QA_ABSTAIN_GATE": "jev"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--kg-dir", default="../Agent-Graph-Memory/evaluation/results/locomo_kg_cache")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    os.environ.update(CONFIGS[args.config])
    os.environ.setdefault("JEV_CACHE_DIR", "evaluation/results/jev/jev_cache")

    from t2_locomo_rerank import CAT_NAMES, OUT_DIR, _gold, _pick_questions
    from t3_locomo_retrieval import _aggregate
    from run_eval import get_answer, score_qa  # evaluation/LoComo (path added by t2)

    from evaluation.answer_format_profiles import get_answer_format
    from multi_agent_kg.core import LLMConfig
    from multi_agent_kg.core.config import RetrievalConfig
    from multi_agent_kg.core.kg_operations import load_governed_kg
    from multi_agent_kg.core.qa_orchestrator import QAOrchestrator
    from multi_agent_kg.core.vector_index import KGVectorStore

    sample = next(s for s in json.loads(Path("evaluation/LoComo/data/locomo10.json").read_text())
                  if s["sample_id"] == args.sample)
    questions = _pick_questions(sample["qa"], args.per_category)
    gkg = load_governed_kg(str(Path(args.kg_dir) / args.sample / "governed_kg.json"))
    retrieval = RetrievalConfig()
    store = KGVectorStore.load_dir(str(OUT_DIR / f"vectors_{args.sample}"), gkg, model=retrieval.embedding_model)
    if store is None:
        store = KGVectorStore(model=retrieval.embedding_model)
        store.build(gkg)
    org_chart = gkg.org_chart
    if not org_chart.domains:
        # Cached LoCoMo KGs carry no org chart; mirror run_eval's non-LLM single-domain fallback.
        from multi_agent_kg.core.domain_experts import OrgChart
        from multi_agent_kg.core.governance import Domain

        org_chart = OrgChart(domains=[Domain(
            domain_id="general", label="General", description="All entities (single-domain fallback).",
            entity_ids=set(gkg.kg.entities.keys()), relation_schema={})])
    orch = QAOrchestrator(org_chart=org_chart, full_kg=gkg.kg, llm_config=LLMConfig(model=os.environ["LLM_DEFAULT_MODEL"]),
                          vector_store=store, retrieval_config=retrieval,
                          answer_format=get_answer_format("locomo"))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        preds = list(pool.map(lambda qa: get_answer(orch, qa["question"], qa["category"]), questions))
    wall = time.perf_counter() - started
    rows = [{"question": qa["question"], "gold": _gold(qa), "category": qa["category"],
             "arm": {"pred": p, "score": score_qa(p, _gold(qa), qa["category"])}}
            for qa, p in zip(questions, preds)]
    result = {"config": args.config, "env": CONFIGS[args.config], "n": len(rows), "wall_s": round(wall, 1),
              "scores": _aggregate(rows, "arm"), "rows": rows}
    (OUT_DIR / f"t4_{args.sample}_{args.config}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
