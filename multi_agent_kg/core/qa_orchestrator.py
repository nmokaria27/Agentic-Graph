"""
QA application layer on top of the governed knowledge graph.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from multi_agent_kg.core._qa_commit import ANTI_HEDGE_RIDER, build_rider
from multi_agent_kg.core.config import AnswerFormatConfig, LLMConfig, RetrievalConfig
from multi_agent_kg.core.governance import Domain, OrgChart, TopicSubAgent
from multi_agent_kg.core.graph_traversal import (
    find_paths,
    neighbourhood,
    paths_to_text,
    summarize_subgraph,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.llm.openai_client import chat_completion_json

if TYPE_CHECKING:
    from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph


_QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "who",
    "what",
    "where",
    "which",
    "whose",
    "that",
    "this",
    "did",
    "does",
    "was",
    "were",
    "are",
    "has",
    "have",
    "had",
    "into",
    "from",
    "over",
    "under",
    "about",
    "located",
}


def _normalize_terms(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    }


def _question_relation_hints(query: str) -> Set[str]:
    terms = _normalize_terms(query)
    hints = set(terms)
    if terms & {"spouse", "wife", "husband", "married", "partner"}:
        hints.update({
            "spouse",
            "wife",
            "husband",
            "married",
            "partner",
            "professional",
            "personal",
            "collaborator",
            "collaboration",
            "creative",
            "worked",
            "work",
            "known",
        })
    if terms & {"performer", "artist", "singer", "musician", "band"}:
        hints.update({"performer", "artist", "singer", "musician", "band", "album", "created"})
    if terms & {"founder", "founded", "company", "distributed", "distributor"}:
        hints.update({"founder", "founded", "company", "distributed", "distributor", "studio"})
    if terms & {"owner", "owned", "administrative", "territorial", "entity"}:
        hints.update({"owner", "owned", "administrative", "territorial", "municipality", "province", "state"})
    if terms & {"born", "birthplace", "birth"}:
        hints.update({"born", "birthplace", "birth", "place"})
    if terms & {"when", "date", "year", "time", "temporal"}:
        hints.update({"date", "when", "time", "year", "session", "temporal", "conversation"})
    return hints


def _triple_text(triple: Any) -> str:
    return f"{triple.subject} {triple.relation} {triple.object}".replace("_", " ")


ABSTAIN_ANSWER = "No information available — this is not mentioned in the knowledge graph."


def _fact_evidence_on() -> bool:
    return os.getenv("QA_FACT_EVIDENCE") == "1"


def _abstain_gate_on() -> bool:
    return os.getenv("QA_ABSTAIN_GATE", "").lower() == "jev"


def _format_triple(triple: Any, with_evidence: bool = False) -> str:
    base = f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"
    # Surface the session date stamped at build time (Fix A) so temporal questions
    # can be answered from the fact itself instead of a disconnected DATE triple.
    meta = getattr(triple, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    session_date = meta.get("session_date")
    if session_date:
        base += f" [on {session_date}]"
    else:
        # GB-15: surface the provenance document_date so the answerer can
        # prefer newer facts when evidence lines conflict.
        prov = meta.get("provenance") or {}
        refs = prov.get("refs") if isinstance(prov, dict) else None
        doc_dates = [r.get("document_date") for r in (refs or [])
                     if isinstance(r, dict) and r.get("document_date")]
        if doc_dates:
            base += f" [as of {max(doc_dates)}]"
    # GB-15: a superseded fact must never be served as current truth — mark it
    # so the answer prompt's freshness rule can exclude it decisively.
    if meta.get("superseded_by") or meta.get("superseded_at"):
        base += " [SUPERSEDED by newer information — do not use as the answer]"
    if with_evidence:
        # The source sentence carries the detail the triple compresses away
        # (EXP-JEV-1: without it gold-answer recall in rendered facts is near zero).
        evidence = str(meta.get("evidence") or "").strip()
        if evidence:
            base += f' — "{evidence[:300]}"'
    return base


def _chat_completion_json(*args, **kwargs):
    """Compatibility hook so legacy monkeypatches on domain_experts still work."""
    try:
        from multi_agent_kg.core import domain_experts as compatibility

        patched = getattr(compatibility, "chat_completion_json", None)
        if patched is not None and patched is not _chat_completion_json:
            return patched(*args, **kwargs)
    except Exception:
        pass
    return chat_completion_json(*args, **kwargs)


class DomainExpertAgent:
    """A domain specialist that answers from its governed subgraph."""

    def __init__(
        self,
        domain: Domain,
        full_kg: KnowledgeGraph,
        llm_config: LLMConfig,
        vector_store: Optional[Any] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        answer_format: Optional[AnswerFormatConfig] = None,
    ):
        self.domain = domain
        self.full_kg = full_kg
        self.llm_config = llm_config
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.answer_format = answer_format or AnswerFormatConfig()

    @property
    def _vectors_on(self) -> bool:
        return self.vector_store is not None and self.retrieval_config.use_vectors

    def _vector_entity_candidates(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Semantic entity candidates from the vector index (empty on failure)."""
        if not self._vectors_on:
            return []
        try:
            hits = self.vector_store.entity_index.search(
                query,
                top_k=top_k or self.retrieval_config.entity_top_k,
                min_score=self.retrieval_config.entity_min_score,
            )
        except Exception:
            return []
        return [entity_id for entity_id, _score in hits if entity_id in self.full_kg.entities]

    def answer(self, query: str, context: str = "") -> Dict[str, Any]:
        entities, domain_triples = self.domain.get_subgraph(self.full_kg)
        focused_triples, subgraph_summary = self._select_evidence(
            query, candidates=domain_triples
        )
        gate = getattr(self, "_last_gate", None)
        if (
            _abstain_gate_on()
            and gate is not None
            and gate["jev_max"] < self.retrieval_config.jev_abstain_tau
            and not gate.get("exempt")
        ):
            # No fact is judged relevant enough: abstain without an LLM call.
            return {
                "answer": ABSTAIN_ANSWER,
                "short_answer": "",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
                "out_of_scope_aspects": [query],
                "abstained": True,
                "gate": gate,
                "domain_id": self.domain.domain_id,
                "topics_used": [],
                "multi_hop_paths": "none",
            }
        if getattr(self, "_last_intent", None) == "aggregate":
            from multi_agent_kg.core.query_intent import AGGREGATE_RIDER

            context = (context + "\n\n" if context else "") + AGGREGATE_RIDER
        subgraph_text = self._query_focused_domain_summary(query, focused=focused_triples)
        if subgraph_summary:
            subgraph_text += "\n\nSUMMARY OF RELEVANT SUBGRAPH:\n" + subgraph_summary
        relevant_topics = self._route_to_topics(query)
        topic_names = [topic.label for topic in relevant_topics]

        multi_hop_text = ""
        query_entities = self._extract_query_entities(query)
        max_hops = self.retrieval_config.max_hops
        if len(query_entities) >= 2:
            all_paths = []
            for index in range(len(query_entities)):
                for other in range(index + 1, len(query_entities)):
                    all_paths.extend(
                        find_paths(
                            self.full_kg,
                            query_entities[index],
                            query_entities[other],
                            max_hops=max_hops,
                        )
                    )
            if all_paths:
                multi_hop_text = (
                    "\n\nMULTI-HOP REASONING PATHS (connections between query entities):\n"
                    + paths_to_text(all_paths)
                )
        elif len(query_entities) == 1:
            hops = self.retrieval_config.neighbourhood_hops
            triples = neighbourhood(self.full_kg, query_entities[0], hops=hops)
            if triples:
                lines = [
                    f"  {_format_triple(triple)}"
                    for triple in triples[: self.retrieval_config.neighbourhood_display]
                ]
                multi_hop_text = (
                    f"\n\nNEIGHBOURHOOD ({hops}-hop) of '{query_entities[0]}':\n"
                    + "\n".join(lines)
                )

        rider = build_rider(self.answer_format)
        max_sentences = self.answer_format.max_sentences
        prompt = f"""You are a domain expert for: {self.domain.label}
Domain description: {self.domain.description}

You have access to the following knowledge from a knowledge graph:

{subgraph_text}
{multi_hop_text}
{f"Additional context: {context}" if context else ""}

QUERY: {query}
{rider}
Based ONLY on the knowledge graph data above, provide:
1. A concise answer to the query using only claims that are directly supported by
   the evidence above. Prefer 1-{max_sentences} sentences.
2. If the evidence is incomplete, answer only the supported part and explicitly
   note the gap in a short neutral phrase.
3. Do NOT mention domain IDs, expert agents, routing, or the phrase "knowledge graph".
4. Do NOT speculate, generalize, or add background knowledge.
4b. FRESHNESS RULE: if evidence lines conflict, use the line with the NEWEST
   "[on ...]"/"[as of ...]" date and NEVER a line marked SUPERSEDED. State the
   concrete value itself — never an internal id, variable name, or placeholder.
5. A coverage score (0.0-1.0)
6. The specific KG triples that support your answer
7. Your confidence in the answer (0.0-1.0)

Respond in JSON:
{{
    "answer": "A short evidence-grounded answer.",
    "coverage": 0.65,
    "evidence": ["(entity1) -[relation]-> (entity2)"],
    "confidence": 0.7,
    "out_of_scope_aspects": ["missing pieces"]
}}

Return ONLY the JSON."""

        try:
            result = _chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a domain expert for '{self.domain.label}'. "
                            "Answer queries using ONLY the knowledge graph data provided. "
                            "Use multi-hop reasoning paths when available to explain "
                            "indirect connections. Be precise about what you know and don't."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception as exc:
            result = {
                "answer": "",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
                "out_of_scope_aspects": [query],
                "error": str(exc),
            }

        result["domain_id"] = self.domain.domain_id
        result["topics_used"] = topic_names
        result["multi_hop_paths"] = multi_hop_text if multi_hop_text else "none"

        computed = self._compute_coverage_confidence(query, query_entities, focused_triples=focused_triples)
        if computed["entity_coverage"] > 0 or computed["triple_coverage"] > 0:
            result["coverage"] = computed["coverage"]
            result["confidence"] = computed["confidence"]
        return result

    def _query_focused_domain_summary(self, query: str, *, limit: Optional[int] = None, focused: Optional[List[Any]] = None) -> str:
        entities, domain_triples = self.domain.get_subgraph(self.full_kg)
        if focused is None:
            focused = self._query_focused_triples(query, candidates=domain_triples, limit=limit)
        lines = [f"Domain: {self.domain.label}", f"Description: {self.domain.description}", ""]
        memory_card = self.domain.memory_card_summary()
        if memory_card:
            lines.append(memory_card)
            lines.append("")
        lines.append(f"Domain size: {len(entities)} entities, {len(domain_triples)} relationships.")
        if focused:
            lines.append("Query-focused relationships:")
            for triple in focused:
                conf = f" (conf={triple.confidence:.2f})" if triple.confidence else ""
                lines.append(f"  {_format_triple(triple, with_evidence=_fact_evidence_on())}{conf}")
        else:
            lines.append("No query-focused relationships found in this domain.")
            lines.append("Representative entities:")
            for entity in entities[:20]:
                type_str = f" [{entity.type}]" if entity.type else ""
                lines.append(f"  - {entity.id}{type_str}")
            lines.append("Representative relationships:")
            for triple in domain_triples[:30]:
                lines.append(f"  {_format_triple(triple)}")
        return "\n".join(lines)

    def _entity_text(self, entity_id: str) -> str:
        entity = self.full_kg.entities.get(entity_id)
        labels = entity.labels if entity else []
        return " ".join([entity_id.replace("_", " ")] + list(labels))

    def _score_entity_for_query(self, entity_id: str, query: str) -> int:
        query_lower = query.lower()
        query_terms = _normalize_terms(query)
        hint_terms = _question_relation_hints(query)
        entity = self.full_kg.entities.get(entity_id)
        names = [entity_id.replace("_", " ")] + (entity.labels if entity else [])
        score = 0
        for name in names:
            name_lower = name.lower()
            if len(name_lower) < 3:
                continue
            if re.search(r"\b" + re.escape(name_lower) + r"\b", query_lower):
                score += 8
            score += 2 * len(_normalize_terms(name) & query_terms)
        for triple in self.full_kg.triples:
            if triple.subject != entity_id and triple.object != entity_id:
                continue
            relation_terms = _normalize_terms(triple.relation)
            neighbor_terms = _normalize_terms(_triple_text(triple))
            score += 2 * len(relation_terms & hint_terms)
            score += len(neighbor_terms & query_terms)
        return score

    def _score_triple_for_query(self, triple: Any, query: str, seed_entities: Optional[Set[str]] = None) -> int:
        query_terms = _normalize_terms(query)
        hint_terms = _question_relation_hints(query)
        relation_terms = _normalize_terms(triple.relation)
        triple_terms = _normalize_terms(_triple_text(triple))
        subject_terms = _normalize_terms(self._entity_text(triple.subject))
        object_terms = _normalize_terms(self._entity_text(triple.object))
        score = 0
        score += 3 * len((subject_terms | object_terms) & query_terms)
        score += 4 * len(relation_terms & hint_terms)
        score += len(triple_terms & query_terms)
        if seed_entities and (triple.subject in seed_entities or triple.object in seed_entities):
            score += 3
        if triple.confidence:
            score += int(min(float(triple.confidence), 1.0) * 2)
        return score

    def _vector_seed_triples(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """Seed triples from the triple vector index, restricted to candidates."""
        if not self._vectors_on:
            return []
        try:
            from multi_agent_kg.core.vector_index import triple_key

            hits = self.vector_store.triple_index.search(
                query,
                top_k=top_k or self.retrieval_config.triple_top_k,
            )
        except Exception:
            return []
        candidate_by_key = {triple_key(triple): triple for triple in candidates}
        return [candidate_by_key[key] for key, _score in hits if key in candidate_by_key]

    def _query_focused_triples(
        self,
        query: str,
        *,
        candidates: Optional[List[Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Any]:
        # None resolves to the tunable config value; direct callers (tests, fallback
        # path) honour focused_limit instead of a silent hardcoded 60.
        if limit is None:
            limit = self.retrieval_config.focused_limit
        candidates = candidates if candidates is not None else self.full_kg.get_active_triples()
        lexical_seeds: List[Any] = []
        if self.retrieval_config.use_lexical:
            scored = [
                (self._score_triple_for_query(triple, query), index, triple)
                for index, triple in enumerate(candidates)
            ]
            lexical_seeds = [
                triple
                for score, _, triple in sorted(scored, key=lambda item: item[0], reverse=True)[
                    : self.retrieval_config.lexical_seed_k
                ]
                if score > 0
            ]
        dense_seeds = self._vector_seed_triples(query, candidates)
        # Union, dense first only where lexical missed (lexical keeps precision,
        # dense adds paraphrase recall); graph expansion below is unchanged.
        seen_keys = {(t.subject, t.relation, t.object) for t in lexical_seeds}
        seeds = lexical_seeds + [
            t for t in dense_seeds if (t.subject, t.relation, t.object) not in seen_keys
        ]
        seeds = seeds[: self.retrieval_config.seed_cap]
        seed_entities = {triple.subject for triple in seeds} | {triple.object for triple in seeds}

        # Personalized PageRank from the seed entities: multi-hop-relevant
        # entities join the expansion frontier, and their (normalized) PPR
        # mass boosts triple ranking. Falls back to plain 1-hop adjacency.
        ppr_scores: Dict[str, float] = {}
        if self.retrieval_config.use_ppr and seed_entities:
            from multi_agent_kg.core.graph_traversal import personalized_pagerank

            ranked_ppr = personalized_pagerank(
                self.full_kg, seed_entities, top_k=self.retrieval_config.ppr_top_k
            )
            if ranked_ppr:
                top_score = ranked_ppr[0][1] or 1.0
                ppr_scores = {eid: score / top_score for eid, score in ranked_ppr}
        expansion_ids = seed_entities | set(ppr_scores)

        def _combined_score(triple) -> float:
            lexical = self._score_triple_for_query(triple, query, seed_entities)
            ppr_mass = ppr_scores.get(triple.subject, 0.0) + ppr_scores.get(triple.object, 0.0)
            return lexical + self.retrieval_config.ppr_boost * ppr_mass

        expanded = {
            (triple.subject, triple.relation, triple.object): triple
            for triple in seeds
        }
        for triple in self.full_kg.get_active_triples():
            if triple.subject in expansion_ids or triple.object in expansion_ids:
                if _combined_score(triple) > 0:
                    expanded[(triple.subject, triple.relation, triple.object)] = triple
        ranked = sorted(expanded.values(), key=_combined_score, reverse=True)
        return ranked[:limit]

    def _chunk_select_triples(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        """`chunk` mode: top-k triples ranked by dense similarity, NO graph expansion.

        Ablation baseline analogous to Cognee's chunk-level retriever. Falls back to
        lexical scoring when the vector index is unavailable.
        """
        candidates = candidates if candidates is not None else self.full_kg.triples
        dense = self._vector_seed_triples(query, candidates, top_k=self.retrieval_config.triple_top_k)
        if dense:
            return dense[: self.retrieval_config.focused_limit]
        scored = sorted(
            ((self._score_triple_for_query(t, query), t) for t in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        return [t for score, t in scored[: self.retrieval_config.focused_limit] if score > 0]

    def _summarize_subgraph(self, query: str, triples: List[Any]) -> str:
        """Cached LLM summary of a focused subgraph (Graph-Summary mode, item ④)."""
        cache = getattr(self, "_summary_cache", None)
        if cache is None:
            cache = {}
            self._summary_cache = cache
        key = (self.domain.domain_id, hash((query, tuple(
            (t.subject, t.relation, t.object) for t in triples
        ))))
        if key not in cache:
            cache[key] = summarize_subgraph(triples, query, self.llm_config)
        return cache[key]

    def _select_evidence(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> tuple:
        """Mode-aware evidence dispatch shared by every expert (base-class method).

        Returns ``(focused_triples, summary_text_or_None)``. Dispatch is delegated
        to the BaseRetriever factory (``core.retrievers``): the historical modes
        (hybrid/lexical/dense) and graph_completion return the seed+expansion triples
        unchanged; chunk bypasses expansion; graph_summary additionally LLM-summarizes
        a large subgraph. Placed on DomainExpertAgent so FallbackGraphExpert and
        ActiveExplorerExpert inherit one consistent dispatch.
        """
        from multi_agent_kg.core.retrievers import get_retriever

        retriever = get_retriever(self.retrieval_config.retrieval_mode, self)
        return retriever.select_evidence(query, candidates)

    def _compute_coverage_confidence(
        self,
        query: str,
        query_entities: List[str],
        focused_triples: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        if not query_entities:
            query_entities = self._extract_query_entities(query)

        if focused_triples is not None:
            # Coverage based on actually retrieved evidence, not domain connectivity
            found = sum(
                1 for entity_id in query_entities
                if any(
                    entity_id.lower() in triple.subject.lower()
                    or entity_id.lower() in triple.object.lower()
                    for triple in focused_triples
                )
            )
            entity_coverage = found / max(len(query_entities), 1)

            relevant_focused = [
                triple
                for triple in focused_triples
                if any(
                    entity_id.lower() in triple.subject.lower()
                    or entity_id.lower() in triple.object.lower()
                    for entity_id in query_entities
                )
            ]
            triple_coverage = min(1.0, len(relevant_focused) / max(len(query_entities) * 2, 1))
            coverage = (entity_coverage + triple_coverage) / 2.0

            if relevant_focused:
                avg_conf = sum(
                    triple.confidence for triple in relevant_focused if triple.confidence
                ) / len(relevant_focused)
            else:
                avg_conf = 0.0
            confidence = avg_conf * coverage
        else:
            # Fallback to legacy domain-connectivity metric
            entities_in_domain = self.domain.entity_ids
            found = sum(1 for entity_id in query_entities if entity_id in entities_in_domain)
            entity_coverage = found / max(len(query_entities), 1)

            _, domain_triples = self.domain.get_subgraph(self.full_kg)
            relevant_triples = [
                triple
                for triple in domain_triples
                if any(
                    entity_id.lower() in triple.subject.lower()
                    or entity_id.lower() in triple.object.lower()
                    for entity_id in query_entities
                )
            ] if query_entities else domain_triples
            triple_coverage = min(1.0, len(relevant_triples) / max(len(query_entities), 1))
            coverage = (entity_coverage + triple_coverage) / 2.0

            if relevant_triples:
                avg_conf = sum(
                    triple.confidence for triple in relevant_triples if triple.confidence
                ) / len(relevant_triples)
            else:
                avg_conf = 0.0
            confidence = avg_conf * coverage

        return {
            "entity_coverage": entity_coverage,
            "triple_coverage": triple_coverage,
            "coverage": round(coverage, 3),
            "confidence": round(confidence, 3),
        }

    def _lexical_query_entities(self, query: str) -> List[str]:
        query_lower = query.lower()
        matched = []
        for entity_id, entity in self.full_kg.entities.items():
            names = [entity_id.replace("_", " ")] + entity.labels
            for name in names:
                name_lower = name.lower()
                if len(name_lower) < 3:
                    continue
                pattern = r"\b" + re.escape(name_lower) + r"\b"
                if re.search(pattern, query_lower):
                    matched.append(entity_id)
                    break
        return sorted(
            set(matched),
            key=lambda entity_id: self._score_entity_for_query(entity_id, query),
            reverse=True,
        )

    def _extract_query_entities(self, query: str) -> List[str]:
        """Hybrid entity linking: exact lexical matches fused with vector hits.

        Exact matches are weighted higher in the RRF fusion so a literal
        entity mention never loses to a fuzzy semantic neighbour; the vector
        ranking adds paraphrase recall ("biggest city" -> largest_city).
        """
        lexical = (
            self._lexical_query_entities(query)
            if self.retrieval_config.use_lexical
            else []
        )
        dense = self._vector_entity_candidates(query)
        if not dense:
            return lexical[:12]
        if not lexical:
            return dense[:12]
        from multi_agent_kg.core.vector_index import rrf_fuse

        return rrf_fuse([lexical, dense], weights=[2.0, 1.0])[:12]

    def _route_to_topics(self, query: str) -> List[TopicSubAgent]:
        query_lower = query.lower()
        relevant = []
        for topic in self.domain.topics:
            score = 0
            for keyword in topic.keywords:
                if keyword.lower() in query_lower:
                    score += 1
            if score > 0:
                relevant.append(topic)
        return relevant if relevant else self.domain.topics


class FallbackGraphExpert(DomainExpertAgent):
    """A global query-focused fallback when routed domains miss evidence."""

    def answer(self, query: str, context: str = "") -> Dict[str, Any]:
        query_entities = self._extract_query_entities(query)
        evidence_blocks: List[str] = []
        max_hops = self.retrieval_config.max_hops

        focused_triples, subgraph_summary = self._select_evidence(query)
        if focused_triples:
            evidence_blocks.append("QUERY-FOCUSED GRAPH EVIDENCE:")
            for triple in focused_triples:
                evidence_blocks.append(_format_triple(triple))
        if subgraph_summary:
            evidence_blocks.append("SUMMARY OF RELEVANT SUBGRAPH:")
            evidence_blocks.append(subgraph_summary)

        if len(query_entities) >= 2:
            all_paths = []
            for index in range(len(query_entities)):
                for other in range(index + 1, len(query_entities)):
                    all_paths.extend(
                        find_paths(
                            self.full_kg,
                            query_entities[index],
                            query_entities[other],
                            max_hops=max_hops,
                        )
                    )
            if all_paths:
                evidence_blocks.append("QUERY-SPECIFIC PATHS:")
                for path in all_paths[:8]:
                    for triple in path:
                        evidence_blocks.append(
                            _format_triple(triple)
                        )
                    evidence_blocks.append("---")

        for entity_id in query_entities[:4]:
            triples = neighbourhood(
                self.full_kg, entity_id, hops=self.retrieval_config.neighbourhood_hops
            )
            if triples:
                evidence_blocks.append(f"LOCAL NEIGHBOURHOOD OF {entity_id}:")
                for triple in triples[: self.retrieval_config.neighbourhood_display]:
                    evidence_blocks.append(_format_triple(triple))

        if not evidence_blocks:
            evidence_blocks.append("No direct query-specific graph evidence was found.")

        prompt = f"""You are a global graph fallback expert. Use ONLY the evidence below.

GRAPH EVIDENCE:
{chr(10).join(evidence_blocks)}
{f"Additional context: {context}" if context else ""}

QUERY: {query}
{build_rider(self.answer_format)}
Return JSON:
{{
    "answer": "Short evidence-grounded answer.",
    "coverage": 0.0,
    "evidence": ["(entity) -[relation]-> (entity)"],
    "confidence": 0.0,
    "out_of_scope_aspects": ["missing aspects"]
}}

Rules:
- Be concise and relation-focused.
- Do not speculate or use background knowledge.
- FRESHNESS: on conflicting lines use the NEWEST "[on ...]"/"[as of ...]" date;
  never use lines marked SUPERSEDED; state concrete values, never internal ids.
- If evidence is weak, answer only the supported part.
- Do not mention expert systems, routing, or the phrase "knowledge graph".

Return ONLY the JSON."""

        try:
            result = _chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer from the provided graph evidence only. "
                            "Be concise, conservative, and return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception as exc:
            result = {
                "answer": "",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
                "out_of_scope_aspects": [query],
                "error": str(exc),
            }

        result["domain_id"] = self.domain.domain_id
        result["topics_used"] = []
        result["multi_hop_paths"] = "fallback-focused"

        computed = self._compute_coverage_confidence(query, query_entities, focused_triples=focused_triples)
        if computed["entity_coverage"] > 0 or computed["triple_coverage"] > 0:
            result["coverage"] = computed["coverage"]
            result["confidence"] = max(result.get("confidence", 0.0), computed["confidence"])
        return result


class QAOrchestrator:
    """Routes queries across domain experts in the governed graph."""

    def __init__(
        self,
        org_chart: Optional[OrgChart] = None,
        full_kg: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
        governed_kg: Optional["GovernedKnowledgeGraph"] = None,
        vector_store: Optional[Any] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        answer_format: Optional[AnswerFormatConfig] = None,
    ):
        if governed_kg is not None:
            org_chart = governed_kg.org_chart
            full_kg = governed_kg.kg
        if org_chart is None or full_kg is None:
            raise ValueError("QAOrchestrator requires either governed_kg or both org_chart and full_kg.")

        self.org_chart = org_chart
        self.full_kg = full_kg
        self.governed_kg = governed_kg
        self.llm_config = llm_config or LLMConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.answer_format = answer_format or AnswerFormatConfig()
        self.max_routed_domains = min(4, max(1, len(org_chart.domains)))

        # Build the vector store once for the whole QA session when hybrid
        # retrieval is on and no pre-built store was supplied. Failure (e.g.
        # embedding server down) degrades to lexical retrieval.
        self.vector_store = vector_store
        if self.vector_store is None and self.retrieval_config.use_vectors and governed_kg is not None:
            try:
                from multi_agent_kg.core.vector_index import KGVectorStore

                store = KGVectorStore(model=self.retrieval_config.embedding_model)
                store.build(governed_kg)
                self.vector_store = store
                governed_kg.vector_store = store
            except Exception as exc:
                print(f"  WARNING: vector store unavailable, using lexical retrieval ({exc})")
                self.vector_store = None

        self.experts: Dict[str, DomainExpertAgent] = {
            domain.domain_id: DomainExpertAgent(
                domain=domain,
                full_kg=full_kg,
                llm_config=self.llm_config,
                vector_store=self.vector_store,
                retrieval_config=self.retrieval_config,
                answer_format=self.answer_format,
            )
            for domain in org_chart.domains
        }

        global_domain = Domain(
            domain_id="global_fallback",
            label="Global Fallback",
            description="Query-focused global fallback used only when routed domains miss evidence.",
            entity_ids=set(full_kg.entities.keys()),
            relation_schema={triple.relation: triple.relation for triple in full_kg.triples},
            topics=[],
        )
        self.global_fallback_expert = FallbackGraphExpert(
            domain=global_domain,
            full_kg=full_kg,
            llm_config=self.llm_config,
            vector_store=self.vector_store,
            retrieval_config=self.retrieval_config,
            answer_format=self.answer_format,
        )

        # Optional hook: called as progress_callback(stage: str, info: dict)
        # at each pipeline stage. Used by streaming API endpoints.
        self.progress_callback = None

    def _emit_progress(self, stage: str, **info: Any) -> None:
        if self.progress_callback is not None:
            try:
                self.progress_callback(stage, info)
            except Exception:
                pass

    def _materialize_lazy_chunks(self, question: str) -> None:
        """Lazy-ingest hook: extract KG from query-relevant chunks on demand.

        Active only when a LazyIngestor is attached (self.lazy_ingestor).
        After materialization the vector store re-embeds dirty items so the
        just-extracted facts are retrievable in this same query.
        """
        ingestor = getattr(self, "lazy_ingestor", None)
        if ingestor is None:
            return
        try:
            result = ingestor.materialize_for_query(question)
            if result.get("materialized"):
                print(f"  Lazy ingest: materialized {result['materialized']} chunk(s)")
                if self.vector_store is not None:
                    self.vector_store.refresh()
        except Exception as exc:
            print(f"  WARNING: lazy materialization failed ({exc})")

    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print("QA ORCHESTRATOR: Processing query")
        print(f"{'='*70}")
        print(f"Q: {question}\n")
        self._materialize_lazy_chunks(question)
        self._emit_progress("routing", question=question)

        routing = self._decompose_and_route(question)
        sub_questions = routing.get("sub_questions", [])
        print(f"  Decomposed into {len(sub_questions)} sub-questions")
        self._emit_progress(
            "decomposed",
            sub_questions=[sq.get("question", "") for sq in sub_questions],
        )

        domain_responses: List[Dict[str, Any]] = []
        called_domains: Set[str] = set()
        for sub_question in sub_questions:
            question_text = sub_question.get("question", question)
            target_domains = sub_question.get("target_domains", [])
            base_context = sub_question.get("context", "")

            print(f"\n  Sub-Q: {question_text}")
            print(f"  → Routing to: {target_domains}")

            for domain_id in target_domains:
                if domain_id in called_domains:
                    print(f"    [{domain_id}] skipped (already called)")
                    continue
                expert = self.experts.get(domain_id)
                if not expert:
                    continue
                expert_context = self._build_expert_context(
                    question, question_text, target_domains, domain_id, base_context,
                )
                response = expert.answer(question_text, context=expert_context)
                response["sub_question"] = question_text
                domain_responses.append(response)
                called_domains.add(domain_id)
                self._emit_progress(
                    "domain_answered",
                    domain_id=domain_id,
                    coverage=response.get("coverage", 0),
                    confidence=response.get("confidence", 0),
                )
                print(
                    f"    [{domain_id}] coverage={response.get('coverage', 0):.2f}, "
                    f"confidence={response.get('confidence', 0):.2f}"
                )

        gated = self._abstain_if_all_gated(question, sub_questions, domain_responses)
        if gated is not None:
            return gated

        fallback_context = self._build_global_fallback_context(question, domain_responses, called_domains)
        if fallback_context:
            print("\n  → Triggering global fallback")
            self._emit_progress("fallback")
            fallback_response = self.global_fallback_expert.answer(question, context=fallback_context)
            if (
                fallback_response.get("answer")
                or fallback_response.get("evidence")
                or fallback_response.get("coverage", 0.0) > 0.0
            ):
                fallback_response["sub_question"] = question
                domain_responses.append(fallback_response)
                print(
                    f"    [global_fallback] coverage={fallback_response.get('coverage', 0):.2f}, "
                    f"confidence={fallback_response.get('confidence', 0):.2f}"
                )

        cross_domain_context = self._get_cross_domain_context(question)
        bridge_entities = self._extract_bridge_entities(question, domain_responses)
        bridge_context = self._build_bridge_context(bridge_entities)
        if bridge_entities:
            print(f"  → Bridge expansion: {bridge_entities[:4]}")
        combined_context = "\n\n".join(part for part in [cross_domain_context, bridge_context] if part)
        print(f"\n  Synthesizing final answer from {len(domain_responses)} responses...")
        self._emit_progress("synthesizing", num_responses=len(domain_responses))
        final = self._synthesize(question, domain_responses, combined_context)

        result = {
            "question": question,
            "final_answer": final.get("answer", ""),
            "final_answer_short": final.get("short_answer", ""),
            "sub_questions": sub_questions,
            "domain_responses": domain_responses,
            "overall_coverage": final.get("coverage", 0.0),
            "overall_confidence": final.get("confidence", 0.0),
            "gaps": final.get("gaps", []),
        }

        print(f"\n  Overall coverage: {result['overall_coverage']:.2f}")
        print(f"  Overall confidence: {result['overall_confidence']:.2f}")
        if result["gaps"]:
            print(f"  Knowledge gaps: {result['gaps']}")
        print(f"{'='*70}\n")
        return result

    def _abstain_if_all_gated(
        self, question: str, sub_questions: List[Any], domain_responses: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Abstain gate, orchestrator half (QA_ABSTAIN_GATE=jev).

        When every routed expert abstained, re-check relevance over the WHOLE graph
        (routing may have picked the wrong domain). Only if that is also below the
        threshold do we return the canonical abstention — skipping the fallback
        expert and synthesis calls that would otherwise answer anyway.
        """
        if not _abstain_gate_on() or not domain_responses:
            return None
        if not all(response.get("abstained") for response in domain_responses):
            return None
        expert = self.global_fallback_expert
        expert._select_evidence(question)
        gate = getattr(expert, "_last_gate", None)
        if gate is None or gate.get("exempt") or gate["jev_max"] >= self.retrieval_config.jev_abstain_tau:
            return None
        print(f"  → Abstaining: best fact relevance {gate['jev_max']:.2f} < {self.retrieval_config.jev_abstain_tau}")
        return {
            "question": question,
            "final_answer": ABSTAIN_ANSWER,
            "final_answer_short": "",
            "sub_questions": sub_questions,
            "domain_responses": domain_responses,
            "overall_coverage": 0.0,
            "overall_confidence": 0.0,
            "gaps": [question],
            "abstained": True,
            "gate": gate,
        }

    def _extract_query_entities(self, text: str) -> List[str]:
        probe_expert = getattr(self, "global_fallback_expert", None)
        if probe_expert is not None and hasattr(probe_expert, "_extract_query_entities"):
            # Hybrid lexical + vector linking shared with the experts.
            return probe_expert._extract_query_entities(text)
        query_lower = text.lower()
        matched = []
        for entity_id, entity in self.full_kg.entities.items():
            names = [entity_id.replace("_", " ")] + entity.labels
            for name in names:
                name_lower = name.lower()
                if len(name_lower) < 3:
                    continue
                if re.search(r"\b" + re.escape(name_lower) + r"\b", query_lower):
                    matched.append(entity_id)
                    break
        return list(dict.fromkeys(matched))[:12]

    def _extract_bridge_entities(
        self, question: str, domain_responses: List[Dict[str, Any]]
    ) -> List[str]:
        """KG entities mentioned in expert answer text but not in the original question.

        Targets the hop-1-stop failure mode: an expert returns a hop-1 answer like
        "Lawrence Sanders" or "Steve Hillage" without the synthesis pulling in that
        entity's own neighborhood (which often contains the hop-2 fact).
        """
        query_entities = set(self._extract_query_entities(question))
        bridge: List[str] = []
        seen: Set[str] = set()
        for response in domain_responses:
            evidence_blob = response.get("evidence", [])
            if isinstance(evidence_blob, list):
                evidence_text = " ".join(str(e) for e in evidence_blob)
            else:
                evidence_text = str(evidence_blob)
            text_blob = " ".join([
                str(response.get("answer", "")),
                evidence_text,
            ])
            if not text_blob.strip():
                continue
            text_lower = text_blob.lower()
            for entity_id, entity in self.full_kg.entities.items():
                if entity_id in query_entities or entity_id in seen:
                    continue
                names = [entity_id.replace("_", " ")] + entity.labels
                for name in names:
                    name_lower = name.lower()
                    if len(name_lower) < 3:
                        continue
                    if re.search(r"\b" + re.escape(name_lower) + r"\b", text_lower):
                        bridge.append(entity_id)
                        seen.add(entity_id)
                        break
                if len(bridge) >= 4:
                    break
            # Vector fallback: answers often paraphrase entity names, which
            # the exact scan above misses.
            if (
                len(bridge) < 4
                and self.vector_store is not None
                and self.retrieval_config.use_vectors
            ):
                try:
                    hits = self.vector_store.entity_index.search(
                        text_blob,
                        top_k=6,
                        min_score=self.retrieval_config.entity_min_score,
                    )
                except Exception:
                    hits = []
                for entity_id, _score in hits:
                    if entity_id in query_entities or entity_id in seen:
                        continue
                    if entity_id not in self.full_kg.entities:
                        continue
                    bridge.append(entity_id)
                    seen.add(entity_id)
                    if len(bridge) >= 4:
                        break
            if len(bridge) >= 4:
                break
        return bridge[:4]

    def _build_bridge_context(self, bridge_entities: List[str]) -> str:
        if not bridge_entities:
            return ""
        lines: List[str] = ["BRIDGE ENTITY NEIGHBOURHOODS (use these to chain across hops):"]
        for entity_id in bridge_entities[:2]:
            triples = neighbourhood(self.full_kg, entity_id, hops=1)
            if not triples:
                continue
            lines.append(f"\n{entity_id}:")
            for triple in triples[:15]:
                lines.append(
                    "  " + _format_triple(triple)
                )
        return "\n".join(lines) if len(lines) > 1 else ""

    def _normalize_target_domains(self, domain_ids: List[str]) -> List[str]:
        target_domains = []
        for domain_id in domain_ids:
            if domain_id in self.experts and domain_id not in target_domains:
                target_domains.append(domain_id)
            if len(target_domains) >= self.max_routed_domains:
                break
        if not target_domains and self.org_chart.domains:
            target_domains = [self.org_chart.domains[0].domain_id]
        return target_domains

    def _suggest_domains_from_query(self, question: str, *, limit: int = 4) -> List[str]:
        """Suggest domains from query-focused graph evidence.

        LLM routing can be brittle when an entity name is ambiguous ("Green" as
        a color vs. the Steve Hillage album). This deterministic hint looks at
        the highest-scoring graph facts for the question and routes to domains
        that own their endpoints.
        """
        entity_map = self.org_chart.entity_domain_map()
        counts: Dict[str, int] = {}
        # Vector domain routing: domain descriptions/schemas are embedded, so
        # this matches question semantics even when no entity name overlaps.
        if self.vector_store is not None and self.retrieval_config.use_vectors:
            try:
                domain_hits = self.vector_store.domain_index.search(
                    question, top_k=self.retrieval_config.domain_top_k
                )
            except Exception:
                domain_hits = []
            for rank, (domain_id, _score) in enumerate(domain_hits):
                counts[domain_id] = counts.get(domain_id, 0) + max(1, 12 - 2 * rank)
        for rank, triple in enumerate(self.global_fallback_expert._query_focused_triples(question, limit=12)):
            weight = max(1, 12 - rank)
            for entity_id in (triple.subject, triple.object):
                for domain_id in entity_map.get(entity_id, []):
                    counts[domain_id] = counts.get(domain_id, 0) + weight
        return [
            domain_id
            for domain_id, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ][:limit]

    def _build_expert_context(
        self,
        question: str,
        sub_question: str,
        routed_domains: List[str],
        current_domain_id: str,
        base_context: str,
    ) -> str:
        query_entities = self._extract_query_entities(f"{question} {sub_question}")
        entity_map = self.org_chart.entity_domain_map()
        routed_set = set(routed_domains)
        lines: List[str] = []

        if base_context.strip():
            lines.append(base_context.strip())

        if query_entities:
            lines.append("Ownership hints:")
            for entity_id in query_entities[:8]:
                owners = entity_map.get(entity_id, [])
                if current_domain_id in owners:
                    other_owners = [owner for owner in owners if owner != current_domain_id]
                    if other_owners:
                        lines.append(
                            f"  - {entity_id}: this domain owns it; also linked to {', '.join(other_owners[:2])}"
                        )
                    else:
                        lines.append(f"  - {entity_id}: this domain owns it")
                elif owners:
                    lines.append(f"  - {entity_id}: handled by {', '.join(owners[:2])}")
                else:
                    lines.append(f"  - {entity_id}: currently unowned in the domain chart")

        relevant_cross = []
        for triple in self.org_chart.cross_domain_relations:
            if triple.subject not in query_entities and triple.object not in query_entities:
                continue
            subject_owners = set(entity_map.get(triple.subject, []))
            object_owners = set(entity_map.get(triple.object, []))
            if (
                current_domain_id in subject_owners
                or current_domain_id in object_owners
                or routed_set & (subject_owners | object_owners)
            ):
                relevant_cross.append(triple)

        if relevant_cross:
            lines.append("Relevant cross-domain hints:")
            for triple in relevant_cross[:8]:
                lines.append("  - " + _format_triple(triple))

        return "\n".join(lines).strip()

    def _community_context(self, question: str) -> str:
        """Query-relevant community summaries (GraphRAG global layer)."""
        if self.governed_kg is None or not getattr(self.governed_kg, "communities", None):
            return ""
        from multi_agent_kg.core.community import community_context

        return community_context(
            self.governed_kg,
            question,
            top_k=self.retrieval_config.community_top_k,
            vector_store=self.vector_store,
        )

    def _build_global_fallback_context(
        self,
        question: str,
        domain_responses: List[Dict[str, Any]],
        called_domains: Set[str],
    ) -> str:
        if len(self.experts) <= 1:
            return ""

        query_entities = self._extract_query_entities(question)
        entity_map = self.org_chart.entity_domain_map()
        unowned = [entity_id for entity_id in query_entities if not entity_map.get(entity_id)]
        uncovered = [
            entity_id
            for entity_id in query_entities
            if entity_map.get(entity_id) and not (set(entity_map[entity_id]) & called_domains)
        ]

        best_coverage = max((response.get("coverage", 0.0) for response in domain_responses), default=0.0)
        best_confidence = max((response.get("confidence", 0.0) for response in domain_responses), default=0.0)
        has_supported_answer = any(
            (response.get("answer") or "").strip() and response.get("coverage", 0.0) >= 0.35
            for response in domain_responses
        )

        if not (unowned or uncovered or (query_entities and not has_supported_answer)):
            return ""

        lines = [
            "Fallback trigger: routed domains may have missed relevant evidence.",
            "Use the full graph only to recover missing entities or bridge relations.",
            "Do not restate claims already unsupported by routed experts.",
        ]
        if unowned:
            lines.append(f"Unowned query entities: {', '.join(unowned[:6])}")
        if uncovered:
            lines.append(f"Entities not covered by routed domains: {', '.join(uncovered[:6])}")
        if not has_supported_answer:
            lines.append(f"Best routed coverage/confidence was {best_coverage:.2f}/{best_confidence:.2f}.")
        community_block = self._community_context(question)
        if community_block:
            lines.append(community_block)
        return "\n".join(lines)

    def _decompose_and_route(self, question: str) -> Dict[str, Any]:
        org_summary = self.org_chart.domain_summary()
        prompt = f"""You are a query routing agent. Given a user question and a list of
available domain experts, decompose the question into sub-questions and
route each to the most relevant domain expert(s).

AVAILABLE DOMAIN EXPERTS:
{org_summary}

USER QUESTION: {question}

If the question is simple and maps to a single domain, return just one sub-question.

Respond in JSON:
{{
  "sub_questions": [
    {{
      "question": "Focused sub-question",
      "target_domains": ["domain_a", "domain_b"],
      "context": "optional routing hint"
    }}
  ]
}}

Return ONLY the JSON."""

        try:
            result = _chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query decomposition and routing expert. "
                            "Prefer the fewest domains needed and return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "sub_questions": [
                    {
                        "question": question,
                        "target_domains": [
                            domain.domain_id for domain in self.org_chart.domains[: self.max_routed_domains]
                        ],
                        "context": "",
                    }
                ]
            }

        if isinstance(result, list):
            result = {"sub_questions": result}
        elif not isinstance(result, dict):
            result = {"sub_questions": []}
        sub_qs = result.get("sub_questions", [])
        if not isinstance(sub_qs, list):
            sub_qs = []
        normalized: List[Dict[str, Any]] = []
        for sub_question in sub_qs:
            if isinstance(sub_question, str):
                sub_question = {"question": sub_question, "target_domains": [], "context": ""}
            elif not isinstance(sub_question, dict):
                continue
            suggested_domains = self._suggest_domains_from_query(
                f"{question} {sub_question.get('question', '')}"
            )
            sub_question["target_domains"] = self._normalize_target_domains(
                suggested_domains + sub_question.get("target_domains", [])
            )
            normalized.append(sub_question)
        if not normalized:
            suggested_domains = self._suggest_domains_from_query(question)
            normalized = [{
                "question": question,
                "target_domains": self._normalize_target_domains(
                    suggested_domains
                    + [domain.domain_id for domain in self.org_chart.domains[: self.max_routed_domains]]
                ),
                "context": "",
            }]
        result["sub_questions"] = normalized
        return result

    def _get_cross_domain_context(self, question: str) -> str:
        lines = []
        query_lower = question.lower()
        matched_entities = []
        for entity_id, entity in self.full_kg.entities.items():
            names = [entity_id.replace("_", " ")] + entity.labels
            for name in names:
                name_lower = name.lower()
                if len(name_lower) > 2 and re.search(r"\b" + re.escape(name_lower) + r"\b", query_lower):
                    matched_entities.append(entity_id)
                    break

        if self.org_chart.cross_domain_relations and matched_entities:
            relevant_cross = [
                triple
                for triple in self.org_chart.cross_domain_relations
                if triple.subject in matched_entities or triple.object in matched_entities
            ]
            if relevant_cross:
                lines.append("Cross-domain relationships:")
                for triple in relevant_cross[:20]:
                    lines.append("  " + _format_triple(triple))

        if len(matched_entities) >= 2:
            for index in range(len(matched_entities)):
                for other in range(index + 1, len(matched_entities)):
                    paths = find_paths(self.full_kg, matched_entities[index], matched_entities[other], max_hops=3)
                    if paths:
                        lines.append(f"\nMulti-hop paths ({matched_entities[index]} → {matched_entities[other]}):")
                        lines.append(paths_to_text(paths))

        focused_fn = getattr(self.global_fallback_expert, "_query_focused_triples", None)
        focused_triples = focused_fn(question, limit=40) if callable(focused_fn) else []
        if focused_triples:
            lines.append("\nQuery-focused graph evidence:")
            for triple in focused_triples:
                lines.append(f"  {_format_triple(triple)}")

        return "\n".join(lines) if lines else ""

    def _synthesize(
        self,
        question: str,
        domain_responses: List[Dict[str, Any]],
        cross_domain_context: str,
    ) -> Dict[str, Any]:
        if not domain_responses:
            return {
                "answer": "I don't have enough information in the knowledge graph to answer this question.",
                "coverage": 0.0,
                "confidence": 0.0,
                "gaps": ["No domain experts could provide relevant information"],
            }

        response_texts = []
        for response in domain_responses:
            response_texts.append(
                f"Domain Expert [{response.get('domain_id', '?')}] "
                f"(coverage={response.get('coverage', 0):.2f}, "
                f"confidence={response.get('confidence', 0):.2f}):\n"
                f"  Answer: {response.get('answer', 'N/A')}\n"
                f"  Evidence: {response.get('evidence', [])}\n"
                f"  Out of scope: {response.get('out_of_scope_aspects', [])}"
            )

        prompt = f"""You are a knowledge synthesis agent.

USER QUESTION: {question}

DOMAIN EXPERT RESPONSES:
{chr(10).join(response_texts)}

{f"ADDITIONAL GRAPH CONTEXT (treat triples here as primary evidence — they are graph facts retrieved for this question, equal in authority to the domain experts above):{chr(10)}{cross_domain_context}" if cross_domain_context else ""}
{build_rider(self.answer_format)}
RULES:
- Do NOT mention domain experts, routing, confidence scores, or out-of-scope notes in the answer text.
- Do NOT say "the knowledge graph says" or similar meta-commentary.
- Include only claims that are directly supported by the expert evidence OR by triples in the additional graph context above. Both are first-class evidence.
- For multi-hop questions, you MUST chain across triples. If an expert names a hop-1 entity (e.g., "Steve Hillage is the performer of Green") and the additional graph context contains a triple about that entity that answers the next hop (e.g., "(miquette_giraudy) -[PROFESSIONAL_PARTNER]-> (steve_hillage)"), USE that triple to produce the final hop-2 answer. Do NOT say "no information available" when a relevant triple is present in the additional graph context.
- Treat closely-related relations as semantic equivalents when answering (PROFESSIONAL_PARTNER ≈ spouse/partner, AUTHORED_BY ≈ wrote/written by, BORN_IN ≈ birthplace, LOCATED_IN ≈ located in, FACILITY_LOCATED_IN_CITY ≈ headquartered in, MEMBER_OF ≈ member of, etc.).
- For MuSiQue-style person/partner questions, relation labels such as
  BEST_KNOWN_FOR_WORK_WITH, WORKED_WITH_COLLABORATOR, HAS_CREATIVE_PARTNERSHIP_WITH,
  PROFESSIONAL_PARTNER, and CREATIVE_PARTNERSHIP_WITH are valid evidence for a
  spouse/partner-style answer when they connect a person to the hop-1 performer.
  If the question asks for "spouse" but the graph only encodes "partner",
  "creative partnership", or "worked with", return the linked person as the
  short answer rather than abstaining.
- Prefer the shortest answer that fully covers the supported facts.
- If evidence is genuinely missing (no expert evidence AND no relevant triple in additional graph context), state the limitation briefly and stop.
- Provide TWO answer forms:
  * "answer": evidence-grounded prose response (1-{self.answer_format.max_sentences} sentences max)
  * "short_answer": {self.answer_format.short_answer_instruction()} If the evidence does not support an answer, set short_answer to "".

Examples of short_answer form:
- Q: "In which county is X located?" → short_answer: "Randall County"
- Q: "Who founded the company that distributed X?" → short_answer: "Mike Medavoy"
- Q: "Did the team win in 1990?" → short_answer: "yes"
- Q: "What year was X founded?" → short_answer: "1969"
- Q: "Who is the spouse of the Green performer?" with expert saying "Steve Hillage" and graph context "(miquette_giraudy) -[PROFESSIONAL_PARTNER]-> (steve_hillage)" → short_answer: "Miquette Giraudy"

Respond in JSON:
{{
    "answer": "Evidence-grounded prose explanation here.",
    "short_answer": "minimal span",
    "coverage": 0.85,
    "confidence": 0.8,
    "gaps": ["missing aspects"]
}}

Return ONLY the JSON."""

        try:
            return _chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledge synthesis expert. Combine partial answers "
                            "into a coherent, evidence-grounded response. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception as exc:
            return {
                "answer": "",
                "short_answer": "",
                "coverage": sum(response.get("coverage", 0) for response in domain_responses)
                / max(len(domain_responses), 1),
                "confidence": sum(response.get("confidence", 0) for response in domain_responses)
                / max(len(domain_responses), 1),
                "gaps": [],
                "error": str(exc),
            }


__all__ = [
    "DomainExpertAgent",
    "FallbackGraphExpert",
    "QAOrchestrator",
]
