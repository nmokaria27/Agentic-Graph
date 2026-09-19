"""Retriever abstraction (Cognee-inspired BaseRetriever pattern).

A retriever turns a query into grounded evidence in three explicit stages:

    retrieve_objects(query)        -> raw graph objects (triples)
    context_from_objects(objects)  -> (focused_triples, summary_text_or_None)
    completion_from_context(...)   -> an answer string

The experts (DomainExpertAgent, FallbackGraphExpert, ActiveExplorerExpert) only
use the first two stages: they keep their own richer synthesis (provenance,
coverage/confidence, critic) as ``completion``. The third stage here is a thin
default used for standalone/testing use, so the abstraction is complete without
forcing every expert through it.

Design rule (matches the rest of the system): retrievers carry NO benchmark
identity and add NO new behaviour in Phase 1 — each subclass simply wraps the
agent's existing, already-tested helper methods so the ``hybrid`` path stays
byte-identical. Phase 2 (graph_completion) overrides ``retrieve_objects`` to add
multi-collection vector seeding + true k-hop expansion.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Evidence is (focused_triples, summary_text_or_None) — the historical
# _select_evidence return shape, preserved exactly.
Evidence = Tuple[List[Any], Optional[str]]


class BaseRetriever:
    """Wraps a host expert and exposes the three-stage retrieval contract.

    ``agent`` must provide ``retrieval_config`` and the helper methods
    ``_query_focused_triples`` / ``_chunk_select_triples`` / ``_summarize_subgraph``
    (all defined on DomainExpertAgent, inherited by the other experts).
    """

    def __init__(self, agent: Any):
        self.agent = agent

    # -- stage 1 -----------------------------------------------------------
    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        """Default: the historical lexical∪dense seed + graph-expansion path."""
        return self.agent._query_focused_triples(
            query,
            candidates=candidates,
            limit=self.agent.retrieval_config.focused_limit,
        )

    # -- stage 2 -----------------------------------------------------------
    def context_from_objects(self, objects: List[Any], query: str) -> Evidence:
        """Default: raw triples, no summary (hybrid/lexical/dense/graph_completion)."""
        return objects, None

    # -- stages 1+2 bundled (the old _select_evidence return) --------------
    def select_evidence(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> Evidence:
        objects = self.retrieve_objects(query, candidates)
        return self.context_from_objects(objects, query)

    # -- stage 3 (thin default; experts override via their own answer()) ---
    def completion_from_context(self, context: str, query: str) -> str:
        """Minimal LLM completion over a prepared context string.

        Experts do not call this — they keep their richer synthesis. Provided so
        the abstraction is whole and usable standalone (e.g. tests, simple CLIs).
        """
        from multi_agent_kg.llm.openai_client import chat_completion_json

        prompt = (
            "Answer the QUESTION using ONLY the CONTEXT. Be concise and do not "
            "add information that is not in the context.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
            'Return JSON: {"answer": "..."}'
        )
        try:
            result = chat_completion_json(
                messages=[
                    {"role": "system", "content": "You answer strictly from the given context."},
                    {"role": "user", "content": prompt},
                ],
                model=self.agent.llm_config.model,
                temperature=0.1,
            )
            if isinstance(result, dict):
                return str(result.get("answer", "")).strip()
        except Exception:
            pass
        return ""


class HybridRetriever(BaseRetriever):
    """Default fused retrieval: lexical ∪ dense seeds + graph expansion.

    Used for ``hybrid`` / ``lexical`` / ``dense`` (the seed mix is gated inside
    ``_query_focused_triples`` by use_lexical/use_vectors) and, in Phase 1, also
    for ``graph_completion`` (Phase 2 promotes that to its own subclass).
    """


class ChunkRetriever(BaseRetriever):
    """``chunk`` mode: top-k triples by dense similarity, NO graph expansion."""

    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        return self.agent._chunk_select_triples(query, candidates)


class GraphCompletionRetriever(HybridRetriever):
    """``graph_completion`` mode (Phase 2): vector-seeded k-hop expansion.

    Closer to Cognee's strongest retriever than the default seed+1-hop path:

      1. Seed from MULTIPLE vector collections (triples + entities), each
         min-max normalized so neither collection dominates, fused by weight.
      2. Map the top seeds to graph nodes and pull a true k-hop neighbourhood
         (``neighbourhood_hops``) around each, bounded by ``neighbourhood_seed_top_k``.
      3. Dedup, re-rank by the agent's query scorer, cap at ``focused_limit``.

    Degrades to the inherited hybrid seed+expand path whenever the vector index is
    unavailable or returns nothing, so it never hard-fails.
    """

    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        agent = self.agent
        cfg = agent.retrieval_config
        candidates = candidates if candidates is not None else agent.full_kg.triples

        if not agent._vectors_on:
            return super().retrieve_objects(query, candidates)

        seeds = self._multi_collection_seeds(query, candidates)
        if not seeds:
            # No vector signal — fall back to the proven hybrid path.
            return super().retrieve_objects(query, candidates)

        from multi_agent_kg.core.graph_traversal import neighbourhood

        # Expand the most relevant seed entities to a true k-hop neighbourhood.
        seed_entities: List[str] = []
        seen_ent = set()
        for triple in seeds:
            for node in (triple.subject, triple.object):
                if node not in seen_ent:
                    seen_ent.add(node)
                    seed_entities.append(node)
        seed_entities = seed_entities[: cfg.neighbourhood_seed_top_k]

        collected = {
            (t.subject, t.relation, t.object): t for t in seeds
        }
        for entity_id in seed_entities:
            for triple in neighbourhood(agent.full_kg, entity_id, hops=cfg.neighbourhood_hops):
                collected[(triple.subject, triple.relation, triple.object)] = triple

        scope = set(seen_ent)
        ranked = sorted(
            collected.values(),
            key=lambda triple: agent._score_triple_for_query(triple, query, scope),
            reverse=True,
        )
        return ranked[: cfg.focused_limit]

    def _multi_collection_seeds(
        self, query: str, candidates: List[Any]
    ) -> List[Any]:
        """Fuse triple- and entity-collection vector hits into seed triples.

        Returns up to ``seed_cap`` candidate triples ranked by normalized,
        weight-fused vector relevance. Restricted to ``candidates`` (the domain
        subgraph) so seeding respects domain ownership.
        """
        agent = self.agent
        cfg = agent.retrieval_config
        store = agent.vector_store
        from multi_agent_kg.core.vector_index import normalize_scores, triple_key

        candidate_by_key = {triple_key(t): t for t in candidates}
        # Map each candidate entity -> the candidate triples it touches (for
        # spreading an entity-collection hit's score onto concrete triples).
        triples_by_entity: dict = {}
        for key, triple in candidate_by_key.items():
            triples_by_entity.setdefault(triple.subject, []).append(key)
            triples_by_entity.setdefault(triple.object, []).append(key)

        fused: dict = {}  # triple_key -> fused relevance

        try:
            triple_hits = store.triple_index.search(query, top_k=cfg.triple_top_k)
        except Exception:
            triple_hits = []
        for key, norm in normalize_scores(triple_hits):
            if key in candidate_by_key:
                fused[key] = max(fused.get(key, 0.0), cfg.seed_weight_triple * norm)

        try:
            entity_hits = store.entity_index.search(
                query,
                top_k=cfg.entity_top_k,
                min_score=cfg.entity_min_score,
            )
        except Exception:
            entity_hits = []
        for entity_id, norm in normalize_scores(entity_hits):
            for key in triples_by_entity.get(entity_id, ()):  # entity_id may be an id
                fused[key] = max(fused.get(key, 0.0), cfg.seed_weight_entity * norm)

        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
        return [candidate_by_key[key] for key in ordered[: cfg.seed_cap]]


class GraphSummaryRetriever(HybridRetriever):
    """``graph_summary`` mode: hybrid retrieval, then LLM-summarize a large subgraph."""

    def context_from_objects(self, objects: List[Any], query: str) -> Evidence:
        summary = None
        if len(objects) > self.agent.retrieval_config.summary_trigger:
            summary = self.agent._summarize_subgraph(query, objects)
        return objects, summary


_JEV_CLIENT: Any = None


def _jev_client() -> Any:
    global _JEV_CLIENT
    if _JEV_CLIENT is None:
        from multi_agent_kg.llm.typesafe_client import JevClient

        _JEV_CLIENT = JevClient(max_questions_per_call=128)
    return _JEV_CLIENT


JEV_RELEVANCE_INSTRUCTION = (
    "Is this fact from the knowledge graph useful evidence for answering the question? Fact: {fact}"
)


class JevCascadeRetriever(HybridRetriever):
    """``jev_cascade``: wide cheap shortlist, then Jev picks the evidence.

    The production pool's recall — not its ranking — was the bottleneck
    (EXP-JEV-1/2), so the shortlist is wide (dense top-``jev_pool`` plus the top
    lexical hits, which keep exact-match and date facts) and one Jev Noul per
    fact selects the top ``jev_top_k``. The best per-fact probability is left on
    ``agent._last_gate`` for the abstain gate. Any Jev failure falls back to
    the hybrid path.
    """

    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        from multi_agent_kg.core.qa_orchestrator import _format_triple

        agent, cfg = self.agent, self.agent.retrieval_config
        agent._last_gate = None
        pool_source = candidates if candidates is not None else agent.full_kg.get_active_triples()
        pool = list(agent._vector_seed_triples(query, pool_source, top_k=cfg.jev_pool))
        seen = {(t.subject, t.relation, t.object) for t in pool}
        lexical = sorted(
            ((agent._score_triple_for_query(t, query), i, t) for i, t in enumerate(pool_source)),
            key=lambda item: (-item[0], item[1]),
        )[: cfg.jev_lexical_k]
        for score, _, triple in lexical:
            key = (triple.subject, triple.relation, triple.object)
            if score > 0 and key not in seen:
                seen.add(key)
                pool.append(triple)
        if not pool:
            return []
        try:
            answers = _jev_client().ask(
                {"question": query},
                {
                    f"f{i}": {
                        "type": "noul",
                        "instructions": JEV_RELEVANCE_INSTRUCTION.format(
                            fact=_format_triple(t, with_evidence=True)
                        ),
                    }
                    for i, t in enumerate(pool)
                },
            )
        except Exception as exc:
            print(f"  WARNING: jev_cascade scoring failed ({exc}); falling back to hybrid retrieval")
            return super().retrieve_objects(query, candidates)
        probs = [float(answers[f"f{i}"]["p"]) for i in range(len(pool))]
        order = sorted(range(len(pool)), key=lambda i: (-probs[i], i))[: cfg.jev_top_k]
        agent._last_gate = {"jev_max": max(probs), "pool": len(pool)}
        return [pool[i] for i in order]


# Mode string -> retriever class.
_RETRIEVERS = {
    "hybrid": HybridRetriever,
    "lexical": HybridRetriever,
    "dense": HybridRetriever,
    "graph_completion": GraphCompletionRetriever,
    "graph_summary": GraphSummaryRetriever,
    "chunk": ChunkRetriever,
    "jev_cascade": JevCascadeRetriever,
}


def get_retriever(mode: str, agent: Any) -> BaseRetriever:
    """Factory: return the retriever for ``mode`` bound to ``agent``.

    Unknown modes fall back to Hybrid (RetrievalConfig already validates the set,
    so this is defensive only).
    """
    return _RETRIEVERS.get(mode, HybridRetriever)(agent)


__all__ = [
    "JevCascadeRetriever",
    "BaseRetriever",
    "HybridRetriever",
    "ChunkRetriever",
    "GraphSummaryRetriever",
    "GraphCompletionRetriever",
    "get_retriever",
]
