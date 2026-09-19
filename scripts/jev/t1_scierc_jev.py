#!/usr/bin/env python3
"""EXP-JEV-1 / T1 — Jev on the SciERC extraction side.

Two modes, both scored with the SAME scorer as EXP-HEADROOM-SCIERC /
EXP-MODEL-LOCAL-2 (``scripts/score_accumulated_scierc_rich.py``):

verify      Take a governed KG built WITHOUT the LLM verification stage
            (``build_governed_scierc.py --skip-verification``) and gate every
            triple (and optionally entity) with Jev Noul/Choice questions,
            sweeping the keep-threshold offline from cached probabilities.
            Compares against the same build WITH LLM verification.

candidates  No LLM at all: n-gram span candidates → Jev Choice (entity type
            or "not an entity"), then same-sentence entity pairs → Jev Choice
            over directional SciERC relations (or none).

Run via the Fireworks/Jev launcher so TYPESAFE_API_KEY is set::

    python scripts/jev/fw_exec.py python scripts/jev/t1_scierc_jev.py verify \
        --kg evaluation/results/jev/scierc10_noverify.json
    python scripts/jev/fw_exec.py python scripts/jev/t1_scierc_jev.py candidates
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from evaluation.adapters.scierc_adapter import SciERCAdapter  # noqa: E402
from multi_agent_kg.llm.typesafe_client import JevClient  # noqa: E402
from scripts.canonicalize_scierc_relations import canonical_relation  # noqa: E402

OUT_DIR = Path("evaluation/results/jev")
DATA_DIR = "evaluation/datasets/scierc"

# Direction-explicit glosses of the SciERC relation labels (A = subject).
REL_GLOSS = {
    "Used-for": "'{a}' is used for, applied to, or serves '{b}'",
    "Feature-of": "'{a}' is a feature, property or characteristic of '{b}'",
    "Hyponym-of": "'{a}' is a kind, type or instance of '{b}'",
    "Part-of": "'{a}' is a part or component of '{b}'",
    "Evaluate-for": "'{a}' is used to evaluate or measure '{b}'",
    "Compare": "'{a}' is compared or contrasted with '{b}'",
    "Conjunction": "'{a}' and '{b}' are mentioned together as parallel items (joined by and/or/as well as)",
}
SYMMETRIC = {"Compare", "Conjunction"}

ENTITY_TYPES = {
    "Task": "a task or problem to solve (e.g. 'machine translation', 'recognition of proper nouns')",
    "Method": "a method, model, algorithm, system or technique",
    "Metric": "an evaluation metric or measure (e.g. 'accuracy', 'F-measure')",
    "Material": "data, a dataset, corpus, language resource or other material",
    "OtherScientificTerm": "another specific scientific term, concept, phenomenon or component",
    "none": "NOT a complete, specific entity mention: a fragment of a longer phrase, a phrase that "
    "cuts off words, a generic word (e.g. 'method', 'approach', 'system', 'it'), or ordinary words",
}


# ── scoring ──────────────────────────────────────────────────────────


def score(kg_payload: Dict[str, Any], tag: str, max_docs: int, split: str = "test") -> Dict[str, Any]:
    kg = kg_payload.get("knowledge_graph", kg_payload)
    for triple in kg.get("triples", []):
        triple["relation"] = canonical_relation(triple.get("relation"))
    kg_path = OUT_DIR / f"t1_{tag}_kg.json"
    out_path = OUT_DIR / f"t1_{tag}_scores.json"
    kg_path.write_text(json.dumps(kg_payload, default=str))
    subprocess.run(
        [sys.executable, "scripts/score_accumulated_scierc_rich.py", "--kg-path", str(kg_path),
         "--max-docs", str(max_docs), "--split", split, "--output", str(out_path)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return summarize(json.loads(out_path.read_text()))


def summarize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def g(key: str, field: str) -> float:
        return round(metrics.get(key, {}).get(field, 0.0), 3)

    return {
        "ent_f1": g("entity_strict", "f1"),
        "strict_p": g("triple_strict", "precision"),
        "strict_r": g("triple_strict", "recall"),
        "strict_f1": g("triple_strict", "f1"),
        "mapped_f1": g("triple_mapped", "f1"),
        "pred_triples": metrics.get("total_predicted_triples"),
    }


# ── verify mode ──────────────────────────────────────────────────────


def _surface(by_id: Dict[str, Any], triple: Dict[str, Any], side: str) -> str:
    meta = triple.get("metadata", {}) or {}
    text = meta.get(f"original_{side}")
    if text:
        return text
    ent = by_id.get(triple.get(side))
    return (ent.get("labels") or [triple.get(side)])[0] if ent else str(triple.get(side))


def verify(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.kg).read_text())
    kg = payload.get("knowledge_graph", payload)
    triples, entities = kg["triples"], kg["entities"]
    by_id = {e["id"]: e for e in entities}
    docs = {d["id"]: d["text"] for d in SciERCAdapter(f"{DATA_DIR}/test.json", skip_generic=True).to_pipeline_input(max_docs=args.max_docs)}

    # One Jev request per document: all triple + entity questions fan out.
    requests: List[Tuple[Any, Dict[str, Any]]] = []
    for doc_id, text in docs.items():
        questions: Dict[str, Any] = {}
        for i, t in enumerate(triples):
            if t.get("source") != doc_id:
                continue
            a, b = _surface(by_id, t, "subject"), _surface(by_id, t, "object")
            rel = canonical_relation(t.get("relation"))
            gloss = REL_GLOSS.get(rel, "'{a}' " + rel + " '{b}'").format(a=a, b=b)
            questions[f"t{i}"] = {
                "type": "noul",
                "instructions": f"Does the document explicitly state or directly imply that {gloss}?",
            }
        for i, e in enumerate(entities):
            if (e.get("metadata") or {}).get("source_document") != doc_id:
                continue
            questions[f"e{i}"] = {
                "type": "choice",
                "instructions": f"In the document, what is the phrase '{(e.get('labels') or [e['id']])[0]}'?",
                "criteria": ENTITY_TYPES,
            }
        requests.append(({"document": text}, questions))

    jev = JevClient(cache_dir=str(OUT_DIR / "jev_cache"), usage_log=str(OUT_DIR / "jev_usage.jsonl"),
                    max_questions_per_call=128)
    started = time.perf_counter()
    answers: Dict[str, Any] = {}
    for part in jev.ask_many(requests):
        answers.update(part)
    wall = time.perf_counter() - started
    print(f"Jev verify: {jev.stats['calls']} calls, {jev.stats['input_tokens']} input tok, "
          f"${jev.cost_usd():.5f}, wall {wall:.2f}s (cache hits {jev.stats['cache_hits']})")

    results: Dict[str, Any] = {
        "jev": {**jev.stats, "cost_usd": jev.cost_usd(), "wall_s": round(wall, 2),
                "questions": sum(len(q) for _, q in requests)},
        "input_noverify": score(json.loads(json.dumps(payload)), "noverify", args.max_docs),
    }
    for tau in args.taus:
        for gate_entities in (False, True):
            kept_t = [t for i, t in enumerate(triples) if answers.get(f"t{i}", {"p": 1.0})["p"] >= tau]
            kept_e = entities
            if gate_entities:
                kept_e = []
                for i, e in enumerate(entities):
                    ans = answers.get(f"e{i}")
                    if ans is None:
                        kept_e.append(e)
                    elif ans["probabilities"].get("none", 0.0) < 0.5:
                        probs = ans["probabilities"]
                        etype = max((k for k in probs if k != "none"), key=lambda k: probs[k])
                        kept_e.append({**e, "type": etype})
            variant = json.loads(json.dumps(payload))
            variant_kg = variant.get("knowledge_graph", variant)
            variant_kg["triples"], variant_kg["entities"] = kept_t, kept_e
            tag = f"jev_tau{tau}" + ("_ent" if gate_entities else "")
            results[tag] = score(variant, tag, args.max_docs)
    if args.llm_verified_kg and Path(args.llm_verified_kg).exists():
        results["llm_verify"] = score(json.loads(Path(args.llm_verified_kg).read_text()), "llmverify", args.max_docs)

    (OUT_DIR / f"t1_verify_results_{args.max_docs}docs.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


# ── candidates mode (no LLM) ─────────────────────────────────────────

_BREAK = re.compile(r"^(-[LR][RSC]B-|[.,;:!?()\[\]\"'`]|``|''|--)$")
_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with", "by", "from", "as", "at",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "which", "we", "our",
    "it", "its", "their", "has", "have", "can", "also", "not", "such", "than", "into", "using", "based",
}


def span_candidates(sentences: List[List[str]], max_len: int) -> List[Tuple[int, int, int, str]]:
    """(sentence_idx, start, end_exclusive, text) spans that don't cross punctuation."""
    out = []
    for s_idx, toks in enumerate(sentences):
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + max_len) + 1):
                window = toks[i:j]
                if _BREAK.match(window[-1]):
                    break
                if window[0].lower() in _STOP or window[-1].lower() in _STOP:
                    continue
                if not any(ch.isalpha() for ch in "".join(window)):
                    continue
                out.append((s_idx, i, j, " ".join(window)))
    return out


def _split_sentences(text: str) -> List[List[str]]:
    sentences, cur = [], []
    for tok in text.split():
        cur.append(tok)
        if tok == ".":
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


def candidates(args: argparse.Namespace) -> None:
    docs = SciERCAdapter(f"{DATA_DIR}/{args.split}.json", skip_generic=True).to_pipeline_input(max_docs=args.max_docs)
    jev = JevClient(cache_dir=str(OUT_DIR / "jev_cache"), usage_log=str(OUT_DIR / "jev_usage.jsonl"),
                    max_questions_per_call=128, max_workers=16)
    started = time.perf_counter()

    # Stage 1: entity typing over span candidates (one fan-out per document).
    doc_spans = {}
    ent_requests = []
    for doc in docs:
        sentences = _split_sentences(doc["text"])
        spans = span_candidates(sentences, args.max_span)
        doc_spans[doc["id"]] = (sentences, spans)
        questions = {
            f"s{k}": {
                "type": "choice",
                "instructions": f"In the document, the exact phrase '{text}' (sentence {s + 1}) is:",
                "criteria": ENTITY_TYPES,
            }
            for k, (s, _, _, text) in enumerate(spans)
        }
        ent_requests.append(({"document": doc["text"]}, questions))
    ent_answers = jev.ask_many(ent_requests)

    # Stage 2: relation choice for same-sentence, non-overlapping entity pairs.
    rel_criteria = {"none": "no relation between them is stated in the sentence"}
    for rel, gloss in REL_GLOSS.items():
        rel_criteria[f"{rel}|ab"] = gloss.format(a="A", b="B")
        if rel not in SYMMETRIC:
            rel_criteria[f"{rel}|ba"] = gloss.format(a="B", b="A")

    doc_entities: Dict[str, List[Dict[str, Any]]] = {}
    rel_requests, rel_index = [], []
    for doc, answers in zip(docs, ent_answers):
        sentences, spans = doc_spans[doc["id"]]
        kept = []
        for k, (s, i, j, text) in enumerate(spans):
            probs = answers[f"s{k}"]["probabilities"]
            if probs.get("none", 0.0) < args.entity_none_max:
                etype = max((t for t in probs if t != "none"), key=lambda t: probs[t])
                kept.append({"s": s, "i": i, "j": j, "text": text, "type": etype,
                             "p_none": probs.get("none", 0.0)})
        doc_entities[doc["id"]] = kept
        questions = {}
        pairs = []
        for x in range(len(kept)):
            for y in range(x + 1, len(kept)):
                a, b = kept[x], kept[y]
                if a["s"] != b["s"] or not (a["j"] <= b["i"] or b["j"] <= a["i"]):
                    continue
                sentence = " ".join(sentences[a["s"]])
                questions[f"p{len(pairs)}"] = {
                    "type": "choice",
                    "instructions": f"Sentence: \"{sentence}\"\nA = '{a['text']}', B = '{b['text']}'. "
                    "Which relation does the sentence state between A and B?",
                    "criteria": rel_criteria,
                }
                pairs.append((a, b))
        rel_requests.append(({"document": doc["text"]}, questions))
        rel_index.append(pairs)
    rel_answers = jev.ask_many(rel_requests)
    wall = time.perf_counter() - started

    def build(ent_max: float, rel_max: float) -> Dict[str, Any]:
        entities, triples = [], []
        for doc, pairs, answers in zip(docs, rel_index, rel_answers):
            for n, ent in enumerate(doc_entities[doc["id"]]):
                if ent["p_none"] >= ent_max:
                    continue
                entities.append({"id": f"{doc['id']}::{n}", "text": ent["text"], "labels": [ent["text"]],
                                 "type": ent["type"], "metadata": {"source_document": doc["id"]}})
            for k, (a, b) in enumerate(pairs):
                if a["p_none"] >= ent_max or b["p_none"] >= ent_max:
                    continue
                ans = answers[f"p{k}"]
                if ans["choice"] == "none" or ans["probabilities"].get("none", 0.0) >= rel_max:
                    continue
                rel, direction = ans["choice"].split("|")
                subj, obj = (a, b) if direction == "ab" else (b, a)
                triples.append({"subject": subj["text"], "relation": rel, "object": obj["text"],
                                "source": doc["id"], "metadata": {"original_subject": subj["text"],
                                                                  "original_object": obj["text"]}})
        return {"knowledge_graph": {"entities": entities, "triples": triples}}

    # Offline threshold grid over cached probabilities (relation answers are
    # per-pair and isolated, so stricter entity gates only drop pairs).
    grid = {}
    for ent_max in args.entity_grid:
        for rel_max in args.relation_grid:
            tag = f"cand_{args.split}{args.max_docs}_e{ent_max}_r{rel_max}"
            grid[f"e{ent_max}_r{rel_max}"] = score(build(ent_max, rel_max), tag, args.max_docs, args.split)
    best = max(grid, key=lambda k: grid[k]["strict_f1"])
    result = {
        "config": vars(args),
        "candidates": sum(len(v[1]) for v in doc_spans.values()),
        "entities_asked_relations": sum(len(v) for v in doc_entities.values()),
        "pairs_asked": sum(len(p) for p in rel_index),
        "jev": {**jev.stats, "cost_usd": jev.cost_usd(), "wall_s": round(wall, 2)},
        "best_by_strict_f1": {best: grid[best]},
        "grid": grid,
    }
    (OUT_DIR / f"t1_candidates_{args.split}{args.max_docs}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "grid"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--kg", required=True, help="governed KG built with --skip-verification")
    v.add_argument("--llm-verified-kg", help="same build WITH LLM verification, for comparison")
    v.add_argument("--max-docs", type=int, default=10)
    v.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    c = sub.add_parser("candidates")
    c.add_argument("--max-docs", type=int, default=10)
    c.add_argument("--max-span", type=int, default=5)
    c.add_argument("--split", default="test", choices=["train", "dev", "test"])
    c.add_argument("--entity-none-max", type=float, default=0.5, help="loosest entity gate asked")
    c.add_argument("--entity-grid", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05, 0.02])
    c.add_argument("--relation-grid", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05, 0.02])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify(args) if args.mode == "verify" else candidates(args)


if __name__ == "__main__":
    main()
