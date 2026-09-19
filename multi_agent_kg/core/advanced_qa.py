"""
Advanced QA System — next-generation multi-expert reasoning architecture.

Extends the base domain_experts.py with five architectural improvements
inspired by KARMA (NeurIPS 2025), ArG, MiroFish, and SciAgents:

1. Active Graph Exploration — iterative "do I need more info?" loops
   (ArG-inspired: experts don't just look up; they EXPLORE the graph)

2. Self-Reflection Critic — adversarial review of synthesized answers
   (SciAgents-inspired: a critic agent challenges every answer)

3. Multi-Expert Debate — structured debate when experts conflict
   (KARMA-inspired: conflict resolution via LLM debate)

4. Persistent QA Session Memory — cross-turn context accumulation
   (MiroFish-inspired: evolving memory that enriches future queries)

5. Provenance Chain Tracking — every claim maps to KG evidence
   (Novel: full audit trail from answer → atomic claims → triples → source)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │              Advanced QA Orchestrator                    │
    │  ┌──────────┐ ┌──────────┐ ┌────────────┐              │
    │  │ Session   │ │ Query    │ │ Provenance │              │
    │  │ Memory    │ │ Planner  │ │ Tracker    │              │
    │  └────┬─────┘ └────┬─────┘ └─────┬──────┘              │
    │       │            │              │                      │
    │       ▼            ▼              ▼                      │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │           Active Explorer Experts                │    │
    │  │  (iterative graph exploration per domain)        │    │
    │  └──────────────────────┬──────────────────────────┘    │
    │                         │                                │
    │                    ┌────▼────┐                           │
    │                    │ Debate  │ (if experts conflict)     │
    │                    │ Arena   │                           │
    │                    └────┬────┘                           │
    │                         │                                │
    │                    ┌────▼────┐                           │
    │                    │ Synth-  │                           │
    │                    │ esizer  │                           │
    │                    └────┬────┘                           │
    │                         │                                │
    │                    ┌────▼────┐                           │
    │                    │ Critic  │ (self-reflection)         │
    │                    │ Agent   │                           │
    │                    └────┬────┘                           │
    │                         │                                │
    │                    ┌────▼────┐                           │
    │                    │ Final   │ (with provenance)         │
    │                    │ Answer  │                           │
    │                    └─────────┘                           │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from multi_agent_kg.core._qa_commit import ANTI_HEDGE_RIDER, build_rider
from multi_agent_kg.core.knowledge_graph import Entity, KnowledgeGraph, Triple
from multi_agent_kg.core.config import AnswerFormatConfig, LLMConfig
from multi_agent_kg.core.domain_experts import (
    Domain,
    DomainBuilder,
    DomainExpertAgent,
    FallbackGraphExpert,
    OrgChart,
    QAOrchestrator,
    TopicSubAgent,
    find_paths,
    neighbourhood,
    paths_to_text,
)
from multi_agent_kg.core.kg_operations import normalize_entity_name, normalize_for_matching
from multi_agent_kg.core.qa_orchestrator import ABSTAIN_ANSWER, _abstain_gate_on, _format_triple
from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json

if TYPE_CHECKING:
    from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph


# ═══════════════════════════════════════════════════════════════════════════
# 5. Provenance Chain — tracks evidence for every claim
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProvenanceRecord:
    """Tracks the evidence chain for a single claim in an answer."""

    claim: str
    supporting_triples: List[Triple] = field(default_factory=list)
    hop_path: List[Triple] = field(default_factory=list)  # Multi-hop path if indirect
    source_domain: str = ""
    confidence: float = 0.0
    evidence_type: str = "direct"  # direct | inferred | aggregated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "supporting_triples": [
                _format_triple(t) for t in self.supporting_triples
            ],
            "hop_path": [
                _format_triple(t) for t in self.hop_path
            ],
            "source_domain": self.source_domain,
            "confidence": self.confidence,
            "evidence_type": self.evidence_type,
        }


@dataclass
class ProvenanceChain:
    """Full provenance chain for a QA answer."""

    records: List[ProvenanceRecord] = field(default_factory=list)
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0

    def add(self, record: ProvenanceRecord):
        self.records.append(record)
        self.total_claims += 1
        if record.supporting_triples or record.hop_path:
            self.grounded_claims += 1
        else:
            self.ungrounded_claims += 1

    def groundedness_ratio(self) -> float:
        return self.grounded_claims / max(self.total_claims, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "groundedness_ratio": round(self.groundedness_ratio(), 4),
            "records": [r.to_dict() for r in self.records],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Session Memory — persistent cross-turn context
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QATurn:
    """A single QA turn in the session."""
    question: str
    answer: str
    domains_consulted: List[str] = field(default_factory=list)
    entities_discussed: List[str] = field(default_factory=list)
    timestamp: float = 0.0


class SessionMemory:
    """
    Persistent QA session memory.

    Tracks conversation context across QA turns:
    - Previously discussed entities → richer context for follow-up questions
    - Resolved ambiguities → don't re-ask the same clarifications
    - User intent patterns → better routing for subsequent queries
    - Accumulated knowledge → build on prior answers
    """

    def __init__(self, max_turns: int = 50):
        self.turns: List[QATurn] = []
        self.max_turns = max_turns

        # Accumulated context
        self.entity_mention_count: Dict[str, int] = defaultdict(int)
        self.domain_query_count: Dict[str, int] = defaultdict(int)
        self.resolved_entities: Dict[str, str] = {}  # ambiguous name → canonical ID
        self.user_interests: List[str] = []  # Inferred topic interests

    def add_turn(self, turn: QATurn):
        """Record a QA turn and update context."""
        turn.timestamp = time.time()
        self.turns.append(turn)

        # Trim old turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        # Update entity mention counts
        for eid in turn.entities_discussed:
            self.entity_mention_count[eid] += 1

        # Update domain query counts
        for did in turn.domains_consulted:
            self.domain_query_count[did] += 1

    def get_context_for_query(self, query: str) -> str:
        """
        Generate context from session history relevant to a new query.
        Includes: recent conversation summary, frequently discussed entities,
        and resolved ambiguities.
        """
        if not self.turns:
            return ""

        lines = ["SESSION CONTEXT (from prior conversation):"]

        # Recent turns (last 3)
        recent = self.turns[-3:]
        lines.append("\nRecent conversation:")
        for turn in recent:
            lines.append(f"  Q: {turn.question[:100]}")
            lines.append(f"  A: {turn.answer[:150]}...")

        # Frequently discussed entities
        top_entities = sorted(
            self.entity_mention_count.items(),
            key=lambda x: -x[1],
        )[:10]
        if top_entities:
            lines.append(f"\nFrequently discussed entities: "
                        f"{', '.join(e for e, _ in top_entities)}")

        # Resolved ambiguities
        if self.resolved_entities:
            lines.append("\nResolved references:")
            for name, canonical in list(self.resolved_entities.items())[:5]:
                lines.append(f"  '{name}' → {canonical}")

        return "\n".join(lines)

    def get_preferred_domains(self) -> List[str]:
        """Return domains the user queries most (for routing boost)."""
        return [
            did for did, _ in sorted(
                self.domain_query_count.items(), key=lambda x: -x[1]
            )[:3]
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_turns": len(self.turns),
            "top_entities": dict(
                sorted(self.entity_mention_count.items(), key=lambda x: -x[1])[:10]
            ),
            "domain_distribution": dict(self.domain_query_count),
            "resolved_entities": self.resolved_entities,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Active Graph Explorer Expert — iterative exploration
# ═══════════════════════════════════════════════════════════════════════════

class ActiveExplorerExpert(DomainExpertAgent):
    """
    Enhanced domain expert that ACTIVELY explores the graph rather than
    performing a single passive lookup.

    Exploration loop (up to max_iterations):
    1. Retrieve initial subgraph context
    2. Generate preliminary answer
    3. Ask: "What information am I missing? What entities should I explore?"
    4. Expand exploration to suggested entities/paths
    5. Refine answer with new information
    6. Repeat until confident or max iterations reached

    This is inspired by ArG (Active Retrieval-Generation) and implements
    a "think-then-search" pattern for KG-grounded QA.
    """

    def __init__(
        self,
        domain: Domain,
        full_kg: KnowledgeGraph,
        llm_config: LLMConfig,
        max_exploration_rounds: int = 3,
        confidence_threshold: float = 0.7,
        vector_store: Optional[Any] = None,
        retrieval_config: Optional[Any] = None,
        answer_format: Optional[Any] = None,
    ):
        super().__init__(
            domain,
            full_kg,
            llm_config,
            vector_store=vector_store,
            retrieval_config=retrieval_config,
            answer_format=answer_format,
        )
        self.max_exploration_rounds = max_exploration_rounds
        self.confidence_threshold = confidence_threshold

    def answer(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Answer with iterative active exploration.

        Each round:
        1. Generate answer from current evidence
        2. Self-assess: what's missing?
        3. Explore additional graph regions
        4. Refine answer

        Returns enriched response with exploration trace.
        """
        # Initial evidence gathering (from base class logic)
        subgraph_text = self.domain.subgraph_summary(self.full_kg)
        query_entities = self._extract_query_entities(query)

        # Build initial multi-hop context
        multi_hop_text = self._build_multi_hop_context(query_entities)

        # Mode-aware evidence (item ⑤): hybrid/lexical/dense keep the historical broad
        # domain dump; chunk narrows to dense-ranked triples; graph_summary appends an
        # LLM subgraph summary (item ④). graph_completion adds the focused triples.
        extra_evidence = ""
        mode = self.retrieval_config.retrieval_mode
        if mode in {"chunk", "graph_summary", "graph_completion", "jev_cascade"}:
            focused, summary = self._select_evidence(query)
            gate = getattr(self, "_last_gate", None)
            if (
                _abstain_gate_on()
                and gate is not None
                and gate["jev_max"] < self.retrieval_config.jev_abstain_tau
                and not gate.get("exempt")
            ):
                # No fact judged relevant: abstain without any exploration LLM calls.
                return {
                    "domain_id": self.domain.domain_id,
                    "answer": ABSTAIN_ANSWER,
                    "coverage": 0.0,
                    "confidence": 0.0,
                    "evidence": [],
                    "topics_used": [],
                    "multi_hop_paths": "none",
                    "exploration_trace": [],
                    "entities_explored": [],
                    "exploration_rounds": 0,
                    "abstained": True,
                    "gate": gate,
                }
            if getattr(self, "_last_intent", None) == "aggregate":
                from multi_agent_kg.core.query_intent import AGGREGATE_RIDER

                context = (context + "\n\n" if context else "") + AGGREGATE_RIDER
            if mode == "jev_cascade" and focused:
                # Jev-selected facts replace the broad domain dump, with their source sentence.
                subgraph_text = "RELEVANT FACTS:\n" + "\n".join(
                    "  " + _format_triple(t, with_evidence=True) for t in focused
                )
            elif mode == "chunk" and focused:
                subgraph_text = "RELEVANT TRIPLES:\n" + "\n".join(
                    "  " + _format_triple(t) for t in focused
                )
            elif mode == "graph_completion" and focused:
                extra_evidence += "\n\nQUERY-FOCUSED TRIPLES:\n" + "\n".join(
                    "  " + _format_triple(t) for t in focused
                )
            if summary:
                extra_evidence += "\n\nSUMMARY OF RELEVANT SUBGRAPH:\n" + summary

        # Track what we've explored
        explored_entities: Set[str] = set(query_entities)
        all_evidence = subgraph_text + multi_hop_text + extra_evidence
        exploration_trace: List[Dict[str, Any]] = []

        current_answer = ""
        current_confidence = 0.0

        for round_num in range(1, self.max_exploration_rounds + 1):
            # Generate answer from current evidence
            answer_result = self._generate_answer(
                query, all_evidence, context, round_num
            )
            current_answer = answer_result.get("answer", "")
            current_confidence = answer_result.get("confidence", 0.0)

            exploration_trace.append({
                "round": round_num,
                "answer_preview": current_answer[:200],
                "confidence": current_confidence,
                "entities_explored": len(explored_entities),
            })

            # Check if we're confident enough to stop
            if current_confidence >= self.confidence_threshold:
                break

            if round_num >= self.max_exploration_rounds:
                break

            # Self-assessment: what's missing?
            missing = self._assess_gaps(query, current_answer, all_evidence)
            entities_to_explore = missing.get("explore_entities", [])
            questions_to_answer = missing.get("unanswered_aspects", [])

            if not entities_to_explore and not questions_to_answer:
                break  # Nothing more to explore

            # Expand exploration
            new_evidence = self._expand_exploration(
                entities_to_explore, explored_entities
            )
            if new_evidence:
                all_evidence += "\n\nADDITIONAL EVIDENCE (exploration round " + str(round_num) + "):\n" + new_evidence
                explored_entities.update(entities_to_explore)
            else:
                break  # No new info found

        # Compute coverage/confidence
        relevant_topics = self._route_to_topics(query)
        computed = self._compute_coverage_confidence(query, query_entities)

        result = {
            "domain_id": self.domain.domain_id,
            "answer": current_answer,
            "coverage": computed.get("coverage", current_confidence),
            "confidence": max(computed.get("confidence", 0), current_confidence),
            "evidence": answer_result.get("evidence", []),
            "topics_used": [t.label for t in relevant_topics],
            "multi_hop_paths": multi_hop_text if multi_hop_text else "none",
            "exploration_trace": exploration_trace,
            "entities_explored": list(explored_entities),
            "exploration_rounds": len(exploration_trace),
        }

        return result

    def _build_multi_hop_context(self, query_entities: List[str]) -> str:
        """Build multi-hop path context between query entities."""
        multi_hop_text = ""
        if len(query_entities) >= 2:
            all_paths = []
            for i in range(len(query_entities)):
                for j in range(i + 1, len(query_entities)):
                    pths = find_paths(
                        self.full_kg,
                        query_entities[i],
                        query_entities[j],
                        max_hops=self.retrieval_config.max_hops,
                    )
                    all_paths.extend(pths)
            if all_paths:
                multi_hop_text = (
                    "\n\nMULTI-HOP REASONING PATHS:\n"
                    + paths_to_text(all_paths)
                )
        elif len(query_entities) == 1:
            nbr = neighbourhood(
                self.full_kg, query_entities[0], hops=self.retrieval_config.neighbourhood_hops
            )
            if nbr:
                nbr_lines = [
                    "  " + _format_triple(t)
                    for t in nbr[: self.retrieval_config.neighbourhood_display]
                ]
                multi_hop_text = (
                    f"\n\nNEIGHBOURHOOD of '{query_entities[0]}':\n"
                    + "\n".join(nbr_lines)
                )
        return multi_hop_text

    def _generate_answer(
        self, query: str, evidence: str, context: str, round_num: int,
    ) -> Dict[str, Any]:
        """Generate an answer from the current evidence pool."""
        round_note = ""
        if round_num > 1:
            round_note = (
                f"\nThis is exploration round {round_num}. You now have MORE evidence "
                "than before. Revise and improve your answer using ALL available evidence."
            )

        prompt = f"""You are a domain expert for: {self.domain.label}
{self.domain.description}

EVIDENCE FROM KNOWLEDGE GRAPH:
{evidence}

{f"Additional context: {context}" if context else ""}
{round_note}

QUERY: {query}
{build_rider(self.answer_format)}
Based ONLY on the evidence above, provide:
1. A concise answer using only claims directly supported by the evidence above.
   Prefer 1-{self.answer_format.max_sentences} sentences and at most 80 words.
2. If the evidence is incomplete, answer only the supported part and note the gap briefly.
3. Do NOT mention expert agents, domains, routing, or the phrase "knowledge graph".
4. Do NOT speculate or add background knowledge.
5. Do NOT define or explain an entity unless that definition is explicitly present in the evidence.
6. If the evidence only shows a relation such as "Used-for", answer with that relation only.
7. A confidence score (0.0-1.0): how completely can you answer from this evidence?
8. Supporting KG triples
9. Any aspects you CANNOT answer from the available evidence

Return JSON:
{{
    "answer": "A short evidence-grounded answer.",
    "confidence": 0.7,
    "evidence": ["(entity1) -[relation]-> (entity2)", ...],
    "unanswered_aspects": ["aspects not covered by current evidence"]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a domain expert for '{self.domain.label}'. "
                            "Answer using ONLY the provided evidence. "
                            "Be concise and honest about what you cannot answer. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "",
                "confidence": 0.0,
                "evidence": [],
                "unanswered_aspects": [query],
            }

        if not isinstance(result, dict):
            result = {
                "answer": "",
                "confidence": 0.0,
                "evidence": [],
                "unanswered_aspects": [query],
            }
        return result

    def _assess_gaps(
        self, query: str, current_answer: str, current_evidence: str,
    ) -> Dict[str, Any]:
        """
        Self-assessment: identify what information is missing and where to look.
        This is the key "active" component — the expert decides what to explore next.
        """
        prompt = f"""You previously answered a query but were not fully confident.
Analyze what information is MISSING and suggest where to look in the knowledge graph.

QUERY: {query}
YOUR CURRENT ANSWER: {current_answer}

Think about:
1. What aspects of the query are NOT well-covered?
2. What additional entities should we explore in the graph?
3. What relationships might connect the missing pieces?

Return JSON:
{{
    "unanswered_aspects": ["list of aspects not yet addressed"],
    "explore_entities": ["entity names to explore in the graph"],
    "explore_relations": ["relationship types to look for"],
    "reasoning": "Why exploring these would help answer the query"
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research strategist. Identify knowledge gaps "
                            "and suggest what to explore next. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.2,
            )
        except Exception:
            result = {
                "unanswered_aspects": [],
                "explore_entities": [],
                "explore_relations": [],
                "reasoning": "Gap assessment failed; stopping exploration safely.",
            }

        if not isinstance(result, dict):
            result = {
                "unanswered_aspects": [],
                "explore_entities": [],
                "explore_relations": [],
                "reasoning": "Gap assessment returned non-dict; stopping exploration safely.",
            }
        return result

    def _expand_exploration(
        self, entities_to_explore: List[str], already_explored: Set[str],
    ) -> str:
        """Explore additional graph regions around suggested entities."""
        new_evidence_lines = []

        for entity_name in entities_to_explore:
            if entity_name in already_explored:
                continue

            # Try to resolve to a KG entity
            resolved = self._resolve_entity_name(entity_name)
            if not resolved:
                continue

            for eid in resolved[:2]:
                if eid in already_explored:
                    continue

                # Get neighbourhood
                nbr = neighbourhood(self.full_kg, eid, hops=2)
                if nbr:
                    new_evidence_lines.append(f"\nNeighbourhood of '{eid}':")
                    for t in nbr[:20]:
                        new_evidence_lines.append(
                            "  " + _format_triple(t)
                        )

        return "\n".join(new_evidence_lines)

    def _resolve_entity_name(self, name: str) -> List[str]:
        """Fuzzy-resolve an entity name to KG entity IDs.

        Uses aggressive normalization to bridge the gap between user-facing
        names (``HOMA-IR``) and internal entity IDs (``homair``).
        """
        agg = normalize_for_matching(name)
        norm = normalize_entity_name(name)
        matches = []
        for eid, entity in self.full_kg.entities.items():
            candidates_agg = {normalize_for_matching(eid)}
            candidates_std = {normalize_entity_name(eid)}
            for label in entity.labels:
                candidates_agg.add(normalize_for_matching(label))
                candidates_std.add(normalize_entity_name(label))
            # Aggressive exact match first
            if agg in candidates_agg:
                matches.append(eid)
            # Substring match on standard forms as fallback
            elif any(len(norm) >= 3 and (norm in c or c in norm) for c in candidates_std):
                matches.append(eid)
        return matches


# ═══════════════════════════════════════════════════════════════════════════
# 2. Self-Reflection Critic Agent
# ═══════════════════════════════════════════════════════════════════════════

class CriticAgent:
    """
    Adversarial critic that reviews synthesized answers for:
    - Unsupported claims (hallucinations)
    - Logical inconsistencies
    - Missing important information
    - Incorrect entity/relation attribution

    Inspired by SciAgents' Critic pattern and KARMA's Evaluator Agent.
    If the critic identifies serious issues, it triggers re-synthesis.
    """

    def __init__(self, kg: KnowledgeGraph, llm_config: LLMConfig):
        self.kg = kg
        self.llm_config = llm_config

    def critique(
        self,
        question: str,
        answer: str,
        domain_responses: List[Dict[str, Any]],
        kg_evidence: str,
    ) -> Dict[str, Any]:
        """
        Critique a synthesized answer.

        Returns:
            {
                "approved": bool,  # True if answer passes critique
                "issues": [...],
                "severity": "none" | "minor" | "major" | "critical",
                "suggestions": [...],
                "revised_answer": str | None,  # If critical issues, provides revision
            }
        """
        # Format domain expert responses for context
        expert_context = ""
        for resp in domain_responses:
            expert_context += (
                f"\n[{resp.get('domain_id', '?')}] "
                f"(conf={resp.get('confidence', 0):.2f}): "
                f"{resp.get('answer', 'N/A')[:300]}\n"
                f"Evidence: {resp.get('evidence', [])}\n"
            )

        prompt = f"""You are an adversarial CRITIC agent. Your job is to find EVERY flaw
in the synthesized answer below. Be thorough and skeptical.

ORIGINAL QUESTION: {question}

SYNTHESIZED ANSWER TO CRITIQUE:
{answer}

DOMAIN EXPERT RESPONSES (the raw inputs that were synthesized):
{expert_context}

KNOWLEDGE GRAPH EVIDENCE (ground truth):
{kg_evidence}

CHECK FOR THESE ISSUES:
1. HALLUCINATIONS: Claims not supported by any expert response or KG evidence
2. CONTRADICTIONS: Answer contradicts KG evidence or expert responses
3. MISATTRIBUTION: Answer attributes facts to the wrong entity
4. LOGICAL ERRORS: Invalid inferences or reasoning chains
5. MISSING INFO: Important information from experts that was dropped during synthesis
6. HEDGING: Answer is vague where KG has specific information

For each issue found, rate severity:
- minor: Slightly imprecise but not misleading
- major: Factually wrong or significantly misleading
- critical: Fundamentally incorrect, answer should be rewritten

SEVERITY CALIBRATION (follow these rules strictly):
- "missing_info" should be "minor" unless the omission makes the answer factually WRONG
  or dangerously misleading. Summarization that drops non-essential detail is acceptable.
- "misattribution" where the meaning is preserved (e.g., "impacts" vs "affects") should be "minor".
- Reserve "major" for claims that are genuinely wrong, misleading, or contradict the KG.
- Reserve "critical" for answers that are fundamentally incorrect or contain dangerous misinformation.
- If ALL issues are minor, set "approved": true and "overall_severity": "minor".

Return JSON:
{{
    "approved": true/false,
    "overall_severity": "none" | "minor" | "major" | "critical",
    "issues": [
        {{
            "type": "hallucination" | "contradiction" | "misattribution" | "logical_error" | "missing_info",
            "description": "Specific description of the issue",
            "severity": "minor" | "major" | "critical",
            "evidence": "KG triple or expert response that exposes the issue"
        }}
    ],
    "suggestions": ["How to fix each issue"],
    "reasoning": "Overall assessment of answer quality"
}}

Return ONLY the JSON. Be thorough but fair — flag real issues at their true severity.
Summarization that omits non-essential detail is acceptable and should be rated "minor", not "major"."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a skeptical critic. Find every flaw in the answer. "
                            "Be strict but fair. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            return {
                "approved": True,
                "overall_severity": "minor",
                "issues": [],
                "suggestions": [],
                "reasoning": "Critic failed; preserving synthesized answer.",
                "revised_answer": None,
            }

        if not isinstance(result, dict):
            result = {
                "approved": True,
                "overall_severity": "minor",
                "issues": [],
                "suggestions": [],
                "reasoning": "Critic returned non-dict; preserving synthesized answer.",
                "revised_answer": None,
            }

        revised = None
        severity = result.get("overall_severity", "none")
        if severity in ("critical", "major"):
            revised = self._generate_revision(
                question, answer, result.get("issues", []),
                result.get("suggestions", []), kg_evidence,
            )

        result["revised_answer"] = revised
        return result

    def _generate_revision(
        self,
        question: str,
        original_answer: str,
        issues: List[Dict],
        suggestions: List[str],
        kg_evidence: str,
    ) -> str:
        """Generate a revised answer that fixes identified issues."""
        def _fmt_issue(i: Any) -> str:
            if isinstance(i, dict):
                return f"  - [{i.get('severity', '?')}] {i.get('type', '?')}: {i.get('description', '')}"
            return f"  - {str(i)}"
        issues_text = "\n".join(_fmt_issue(i) for i in (issues or []))
        suggestions_text = "\n".join(
            f"  - {s if isinstance(s, str) else str(s)}" for s in (suggestions or [])
        )

        prompt = f"""Revise this answer to fix the identified issues.

QUESTION: {question}

ORIGINAL ANSWER:
{original_answer}

ISSUES FOUND:
{issues_text}

SUGGESTIONS:
{suggestions_text}

KNOWLEDGE GRAPH EVIDENCE (ground truth to use):
{kg_evidence}

Write a CORRECTED answer that:
1. Fixes all identified issues
2. Only makes claims supported by KG evidence
3. Is clear about uncertainty where evidence is limited

Return JSON:
{{"revised_answer": "The corrected answer here..."}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise answer writer. Fix all issues while staying "
                            "strictly grounded in KG evidence. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            return original_answer

        if not isinstance(result, dict):
            return original_answer
        return result.get("revised_answer", original_answer)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Multi-Expert Debate Arena
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DebateArgument:
    """A single argument in a multi-expert debate."""
    domain_id: str
    position: str  # The claim being argued
    evidence: List[str]  # KG triples supporting the position
    confidence: float
    counter_to: Optional[str] = None  # Domain ID being countered


class DebateArena:
    """
    Structured debate between domain experts when their answers conflict.

    Protocol:
    1. Identify conflicts between expert responses
    2. Each conflicting expert presents evidence
    3. Experts counter each other's arguments (up to max_rounds)
    4. Arbiter resolves based on evidence strength and consensus

    Inspired by KARMA's Conflict Resolution Agent.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        max_rounds: int = 2,
        answer_format: Optional[AnswerFormatConfig] = None,
    ):
        self.llm_config = llm_config
        self.max_rounds = max_rounds
        self.answer_format = answer_format or AnswerFormatConfig()

    def detect_conflicts(
        self, domain_responses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between domain expert responses.
        A conflict exists when two experts make contradictory claims
        about the same entity or relationship.
        """
        if len(domain_responses) < 2:
            return []

        # Format responses for conflict detection
        response_summaries = []
        for resp in domain_responses:
            response_summaries.append(
                f"[{resp.get('domain_id', '?')}] "
                f"(conf={resp.get('confidence', 0):.2f}): "
                f"{resp.get('answer', 'N/A')[:400]}"
            )

        prompt = f"""Analyze these domain expert responses and identify CONFLICTS
(cases where experts provide contradictory information about the same thing).

EXPERT RESPONSES:
{chr(10).join(response_summaries)}

A conflict exists when:
- Two experts claim different things about the same entity
- Causal directions disagree (A causes B vs B causes A)
- One expert includes information that contradicts another
- Numerical values or temporal claims disagree

Return JSON:
{{
    "conflicts": [
        {{
            "description": "What the conflict is about",
            "domain_a": "domain_id of first expert",
            "claim_a": "What domain_a claims",
            "domain_b": "domain_id of second expert",
            "claim_b": "What domain_b claims (contradicts claim_a)"
        }}
    ]
}}

If there are no conflicts, return {{"conflicts": []}}.
Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conflict detector. Identify contradictions "
                            "between expert responses. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            return []

        if not isinstance(result, dict):
            return []
        return result.get("conflicts", [])

    def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        domain_responses: Dict[str, Dict[str, Any]],
        kg: KnowledgeGraph,
    ) -> Dict[str, Any]:
        """
        Resolve a conflict through structured debate and evidence evaluation.

        Returns:
            {
                "resolution": str,  # The resolved claim
                "winning_domain": str,  # Which expert's position was adopted
                "reasoning": str,
                "confidence": float,
                "debate_transcript": [...],
            }
        """
        domain_a = conflict.get("domain_a", "")
        domain_b = conflict.get("domain_b", "")
        claim_a = conflict.get("claim_a", "")
        claim_b = conflict.get("claim_b", "")

        resp_a = domain_responses.get(domain_a, {})
        resp_b = domain_responses.get(domain_b, {})

        transcript: List[Dict[str, str]] = []

        # Round 1: Each side presents evidence
        evidence_a = resp_a.get("evidence", [])
        evidence_b = resp_b.get("evidence", [])

        transcript.append({
            "speaker": domain_a,
            "argument": f"I claim: {claim_a}. Evidence: {evidence_a}",
        })
        transcript.append({
            "speaker": domain_b,
            "argument": f"I claim: {claim_b}. Evidence: {evidence_b}",
        })

        # Round 2+: Counter-arguments
        for round_num in range(self.max_rounds):
            # A counters B
            counter_a = self._generate_counter(
                domain_a, claim_a, evidence_a,
                domain_b, claim_b, evidence_b,
                conflict.get("description", ""),
            )
            transcript.append({"speaker": domain_a, "argument": counter_a})

            # B counters A
            counter_b = self._generate_counter(
                domain_b, claim_b, evidence_b,
                domain_a, claim_a, evidence_a + [counter_a],
                conflict.get("description", ""),
            )
            transcript.append({"speaker": domain_b, "argument": counter_b})

        # Arbitration: resolve based on evidence strength
        resolution = self._arbitrate(conflict, transcript, kg)
        resolution["debate_transcript"] = transcript

        return resolution

    def _generate_counter(
        self,
        my_domain: str,
        my_claim: str,
        my_evidence: List[str],
        their_domain: str,
        their_claim: str,
        their_evidence: List[str],
        conflict_desc: str,
    ) -> str:
        """Generate a counter-argument in the debate."""
        prompt = f"""You are domain expert [{my_domain}]. Counter the opposing expert's argument.

CONFLICT: {conflict_desc}
YOUR POSITION: {my_claim}
YOUR EVIDENCE: {my_evidence}
OPPOSING POSITION ({their_domain}): {their_claim}
OPPOSING EVIDENCE: {their_evidence}

Provide a brief counter-argument (2-{self.answer_format.max_sentences} sentences) that:
1. Points out weaknesses in the opposing evidence
2. Strengthens your position with additional reasoning
3. Cites your evidence specifically

Return JSON:
{{"counter_argument": "Your counter-argument here"}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are domain expert [{my_domain}]. Argue your position. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.2,
            )
        except Exception:
            return f"[{my_domain}] maintains its position."

        if not isinstance(result, dict):
            return f"[{my_domain}] maintains its position."
        return result.get("counter_argument", f"[{my_domain}] maintains its position.")

    def _arbitrate(
        self,
        conflict: Dict[str, Any],
        transcript: List[Dict[str, str]],
        kg: KnowledgeGraph,
    ) -> Dict[str, Any]:
        """Impartial arbitration to resolve the debate."""
        transcript_text = "\n".join(
            f"  [{t['speaker']}]: {t['argument']}" for t in transcript
        )

        prompt = f"""You are an impartial ARBITER. Two domain experts have debated a conflict.
Based on the strength of evidence and reasoning, resolve the conflict.

CONFLICT: {conflict.get('description', '')}

DEBATE TRANSCRIPT:
{transcript_text}

Resolve by:
1. Evaluating which side has stronger evidence from the knowledge graph
2. Checking if either side's reasoning is logically flawed
3. Considering whether both positions could be partially correct

Return JSON:
{{
    "resolution": "The resolved factual claim (what is true)",
    "winning_domain": "domain_id of the expert with the stronger position",
    "reasoning": "Why this resolution was chosen",
    "confidence": 0.0-1.0,
    "both_partially_correct": true/false
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an impartial arbiter. Resolve debates fairly based on evidence. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            return {
                "resolution": conflict.get("claim_a", "") or conflict.get("description", ""),
                "winning_domain": conflict.get("domain_a", ""),
                "reasoning": "Arbitration failed; defaulting to the first claim.",
                "confidence": 0.0,
                "both_partially_correct": False,
            }

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Advanced QA Orchestrator — integrates all 5 improvements
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedQAOrchestrator:
    """
    Next-generation QA orchestrator that extends the base QAOrchestrator
    with all five architectural improvements.

    Flow:
    1. Session memory provides context from prior turns
    2. Query decomposition and routing (enhanced with memory)
    3. Active Explorer Experts gather evidence iteratively
    4. Debate Arena resolves conflicts between experts
    5. Synthesizer combines expert responses
    6. Critic Agent reviews and potentially revises the answer
    7. Provenance Tracker builds evidence chain
    8. Session memory is updated

    This creates a self-improving system where:
    - Each QA turn enriches the session context
    - Active exploration finds information passive lookup misses
    - Debate resolves contradictions explicitly
    - The critic catches hallucinations before they reach the user
    - Provenance makes every claim auditable
    """

    def __init__(
        self,
        org_chart: Optional[OrgChart] = None,
        full_kg: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
        governed_kg: Optional["GovernedKnowledgeGraph"] = None,
        max_exploration_rounds: int = 3,
        enable_debate: bool = True,
        enable_critic: bool = True,
        max_critic_revisions: int = 2,
        vector_store: Optional[Any] = None,
        retrieval_config: Optional[Any] = None,
        answer_format: Optional[AnswerFormatConfig] = None,
    ):
        if governed_kg is not None:
            org_chart = governed_kg.org_chart
            full_kg = governed_kg.kg
        if org_chart is None or full_kg is None:
            raise ValueError(
                "AdvancedQAOrchestrator requires either governed_kg or both org_chart and full_kg."
            )
        self.org_chart = org_chart
        self.full_kg = full_kg
        self.governed_kg = governed_kg
        self.llm_config = llm_config or LLMConfig()
        self.answer_format = answer_format or AnswerFormatConfig()
        self.enable_debate = enable_debate
        self.enable_critic = enable_critic
        self.max_critic_revisions = max_critic_revisions
        self.max_routed_domains = min(4, max(1, len(org_chart.domains)))

        from multi_agent_kg.core.config import RetrievalConfig

        self.retrieval_config = retrieval_config or RetrievalConfig()
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

        # Initialize Active Explorer Experts (upgrade #1)
        self.experts: Dict[str, ActiveExplorerExpert] = {}
        for domain in org_chart.domains:
            self.experts[domain.domain_id] = ActiveExplorerExpert(
                domain=domain,
                full_kg=full_kg,
                llm_config=self.llm_config,
                max_exploration_rounds=max_exploration_rounds,
                vector_store=self.vector_store,
                retrieval_config=self.retrieval_config,
                answer_format=self.answer_format,
            )

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

        # Initialize components for improvements #2-5
        self.critic = CriticAgent(full_kg, self.llm_config) if enable_critic else None
        self.debate_arena = (
            DebateArena(self.llm_config, answer_format=self.answer_format)
            if enable_debate
            else None
        )
        self.session_memory = SessionMemory()
        self.provenance_tracker = ProvenanceChain()

        # Optional hook: called as progress_callback(stage: str, info: dict)
        # at each pipeline stage. Used by streaming API endpoints.
        self.progress_callback = None

    def _emit_progress(self, stage: str, **info: Any) -> None:
        if self.progress_callback is not None:
            try:
                self.progress_callback(stage, info)
            except Exception:
                pass

    def _extract_query_entities(self, text: str) -> List[str]:
        import re
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
        return matched

    def query_benchmark_question(self, benchmark_question: Any) -> Dict[str, Any]:
        """Evaluate benchmark questions without leaking memory across independent items."""
        original_memory = self.session_memory
        original_provenance = self.provenance_tracker
        self.session_memory = SessionMemory()
        self.provenance_tracker = ProvenanceChain()
        try:
            return self.query(benchmark_question.question)
        finally:
            self.session_memory = original_memory
            self.provenance_tracker = original_provenance

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the full advanced QA pipeline.

        Returns comprehensive result with exploration traces, debate
        transcripts, critique results, and provenance chains.
        """
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"ADVANCED QA ORCHESTRATOR: Processing query")
        print(f"{'='*70}")
        print(f"Q: {question}\n")

        # Lazy-ingest hook: extract KG from query-relevant chunks on demand,
        # then refresh the vector store so new facts are retrievable now.
        ingestor = getattr(self, "lazy_ingestor", None)
        if ingestor is not None:
            try:
                lazy_result = ingestor.materialize_for_query(question)
                if lazy_result.get("materialized"):
                    print(f"  Lazy ingest: materialized {lazy_result['materialized']} chunk(s)")
                    if self.vector_store is not None:
                        self.vector_store.refresh()
            except Exception as exc:
                print(f"  WARNING: lazy materialization failed ({exc})")

        # ── Step 0: Session context ─────────────────────────────────
        session_context = self.session_memory.get_context_for_query(question)
        if session_context:
            print(f"  [Session] Context from {len(self.session_memory.turns)} prior turns")

        # ── Step 1: Decompose and route ─────────────────────────────
        self._emit_progress("routing", question=question)
        routing = self._decompose_and_route(question, session_context)
        sub_questions = routing.get("sub_questions", [])
        print(f"  Decomposed into {len(sub_questions)} sub-questions")
        self._emit_progress(
            "decomposed",
            sub_questions=[sq.get("question", "") for sq in sub_questions],
        )

        # ── Step 2: Active exploration by domain experts ────────────
        domain_responses: List[Dict[str, Any]] = []
        called_domains: Set[str] = set()
        # step_answers[i] = short answer string from sub-question i (for chain resolution)
        step_answers: List[str] = []

        for sq_idx, sq in enumerate(sub_questions):
            sq_text = sq.get("question", question)
            target_domains = sq.get("target_domains", [])
            sq_context = sq.get("context", "")

            # ── Multi-hop chain: inject prior step answer ──────────
            depends_on = sq.get("depends_on")
            if depends_on is not None and isinstance(depends_on, int) and 0 <= depends_on < len(step_answers):
                prior_answer = step_answers[depends_on]
                if prior_answer:
                    # Replace [answer from step N] placeholder if present, else append
                    import re as _re
                    placeholder = _re.compile(r"\[answer from step \d+\]", _re.IGNORECASE)
                    if placeholder.search(sq_text):
                        sq_text = placeholder.sub(prior_answer, sq_text)
                    else:
                        sq_text = f"{sq_text} (context: {prior_answer})"
                    sq_context = f"Prior step answer: {prior_answer}\n{sq_context}".strip()
                    print(f"  [MultiHop] Step {sq_idx} injected prior answer: {prior_answer[:80]}")

            # Boost preferred domains from session memory
            preferred = self.session_memory.get_preferred_domains()
            for pd in preferred:
                if pd not in target_domains and pd in self.experts:
                    target_domains.append(pd)

            print(f"\n  Sub-Q: {sq_text}")
            print(f"  → Routing to: {target_domains}")

            _sq_responses: List[Dict[str, Any]] = []
            for domain_id in target_domains:
                if domain_id in called_domains:
                    continue

                expert = self.experts.get(domain_id)
                if expert:
                    full_context = f"{sq_context}\n{session_context}" if session_context else sq_context
                    expert_context = QAOrchestrator._build_expert_context(
                        self, question, sq_text, target_domains, domain_id, full_context,
                    )
                    response = expert.answer(sq_text, context=expert_context)
                    response["sub_question"] = sq_text
                    domain_responses.append(response)
                    _sq_responses.append(response)
                    called_domains.add(domain_id)
                    self._emit_progress(
                        "domain_answered",
                        domain_id=domain_id,
                        coverage=response.get("coverage", 0),
                        confidence=response.get("confidence", 0),
                    )

                    rounds = response.get("exploration_rounds", 1)
                    print(f"    [{domain_id}] coverage={response.get('coverage', 0):.2f}, "
                          f"confidence={response.get('confidence', 0):.2f}, "
                          f"exploration_rounds={rounds}")

            # Capture best answer for chained sub-questions
            _best = max(_sq_responses, key=lambda r: r.get("confidence", 0.0), default=None)
            if _best:
                step_answers.append(
                    str(_best.get("short_answer") or _best.get("answer") or "").strip()
                )
            else:
                step_answers.append("")

        gated = QAOrchestrator._abstain_if_all_gated(self, question, sub_questions, domain_responses)
        if gated is not None:
            return gated

        fallback_context = QAOrchestrator._build_global_fallback_context(
            self, question, domain_responses, called_domains,
        )
        if fallback_context:
            print("\n  → Triggering global fallback")
            self._emit_progress("fallback")
            fallback_response = self.global_fallback_expert.answer(
                question, context=fallback_context,
            )
            if (
                fallback_response.get("answer")
                or fallback_response.get("evidence")
                or fallback_response.get("coverage", 0.0) > 0.0
            ):
                fallback_response["sub_question"] = question
                domain_responses.append(fallback_response)
                print(
                    "    [global_fallback] coverage="
                    f"{fallback_response.get('coverage', 0):.2f}, "
                    f"confidence={fallback_response.get('confidence', 0):.2f}"
                )

        # ── Step 3: Debate conflicting responses ────────────────────
        debate_results = []
        if self.debate_arena and len(domain_responses) >= 2:
            conflicts = self.debate_arena.detect_conflicts(domain_responses)
            if conflicts:
                print(f"\n  [Debate] {len(conflicts)} conflict(s) detected!")
                self._emit_progress("debate", num_conflicts=len(conflicts))
                resp_by_domain = {r["domain_id"]: r for r in domain_responses}
                for conflict in conflicts:
                    print(f"    Debating: {conflict.get('description', '?')[:80]}")
                    resolution = self.debate_arena.resolve_conflict(
                        conflict, resp_by_domain, self.full_kg,
                    )
                    debate_results.append({
                        "conflict": conflict,
                        "resolution": resolution,
                    })
                    print(f"    → Resolved: {resolution.get('resolution', '?')[:80]} "
                          f"(winner: {resolution.get('winning_domain', '?')})")

        # ── Step 4: Synthesize ──────────────────────────────────────
        cross_domain_context = self._get_cross_domain_context(question)
        debate_context = self._format_debate_results(debate_results)

        print(f"\n  Synthesizing from {len(domain_responses)} responses...")
        self._emit_progress("synthesizing", num_responses=len(domain_responses))
        synthesized = self._synthesize(
            question, domain_responses, cross_domain_context, debate_context,
        )

        final_answer = synthesized.get("answer", "")

        # ── Step 5: Critic review ───────────────────────────────────
        critic_result = None
        if self.critic:
            kg_evidence = self._get_relevant_evidence(question, final_answer)
            print("  [Critic] Reviewing synthesized answer...")
            self._emit_progress("critic")

            for revision_round in range(self.max_critic_revisions):
                critic_result = self.critic.critique(
                    question, final_answer, domain_responses, kg_evidence,
                )
                severity = critic_result.get("overall_severity", "none")
                approved = critic_result.get("approved", True)
                n_issues = len(critic_result.get("issues", []))

                print(f"    Severity: {severity}, Issues: {n_issues}, "
                      f"Approved: {approved}")

                if approved or severity in ("none", "minor"):
                    break

                # Use revised answer if provided
                if critic_result.get("revised_answer"):
                    final_answer = critic_result["revised_answer"]
                    print(f"    → Answer revised (round {revision_round + 1})")
                else:
                    break

        # ── Step 6: Build provenance chain ──────────────────────────
        provenance = self._build_provenance(final_answer, domain_responses)

        # ── Step 7: Update session memory ───────────────────────────
        discussed_entities = []
        for resp in domain_responses:
            discussed_entities.extend(resp.get("entities_explored", []))

        self.session_memory.add_turn(QATurn(
            question=question,
            answer=final_answer,
            domains_consulted=list(called_domains),
            entities_discussed=discussed_entities,
        ))

        # ── Assemble result ─────────────────────────────────────────
        duration = time.time() - start_time

        result = {
            "question": question,
            "final_answer": final_answer,
            "final_answer_short": synthesized.get("short_answer", ""),
            "sub_questions": sub_questions,
            "domain_responses": domain_responses,
            "overall_coverage": synthesized.get("coverage", 0.0),
            "overall_confidence": synthesized.get("confidence", 0.0),
            "gaps": synthesized.get("gaps", []),
            # Advanced features
            "debate_results": debate_results,
            "critic_result": critic_result,
            "provenance": provenance.to_dict(),
            "session_context": {
                "turn_number": len(self.session_memory.turns),
                "entities_accumulated": len(self.session_memory.entity_mention_count),
            },
            "exploration_summary": {
                "total_rounds": sum(
                    r.get("exploration_rounds", 1) for r in domain_responses
                ),
                "domains_explored": list(called_domains),
            },
            "duration_seconds": round(duration, 2),
        }

        print(f"\n  Overall coverage: {result['overall_coverage']:.2f}")
        print(f"  Overall confidence: {result['overall_confidence']:.2f}")
        print(f"  Provenance: {provenance.grounded_claims}/{provenance.total_claims} grounded")
        print(f"  Duration: {duration:.1f}s")
        if result["gaps"]:
            print(f"  Gaps: {result['gaps']}")
        print(f"{'='*70}\n")

        return result

    # Multi-hop bridge patterns — questions that require chaining two lookups
    _MULTIHOP_PATTERNS = [
        r"\b(who|what|where|when)\b.{0,60}\b(that|which|who)\b",
        r"\b(founded|created|established|owned|acquired|discovered)\b.{0,60}\b(also|later|then|who)\b",
        r"\bcompany\b.{0,40}\b(that|which)\b",
        r"\bperson\b.{0,40}\b(that|who|which)\b",
    ]

    @classmethod
    def _is_multihop(cls, question: str) -> bool:
        """Heuristically detect bridge/chain questions that need multi-hop planning."""
        import re as _re
        q = question.lower()
        return any(_re.search(pat, q) for pat in cls._MULTIHOP_PATTERNS)

    def _decompose_and_route(
        self, question: str, session_context: str,
    ) -> Dict[str, Any]:
        """Enhanced decomposition with session context and multi-hop chain planning."""
        org_summary = self.org_chart.domain_summary()

        session_note = ""
        if session_context:
            session_note = f"""
SESSION CONTEXT (prior conversation — resolve pronouns/references like "it", "that", "the same"):
{session_context}
"""

        multihop_note = ""
        if self._is_multihop(question):
            multihop_note = """
MULTI-HOP DETECTED: This question requires chaining answers. Break it into an
ordered plan where later steps depend on the answers from earlier steps.
Use depends_on to declare which step index (0-based) this step relies on.
Examples:
  Q: "Who founded the company that acquired X?"
  Step 0: "Which company acquired X?" — finds the company name
  Step 1: "Who founded [answer from step 0]?" — uses answer from step 0
"""

        prompt = f"""You are a query planning agent. Decompose the question and route to experts.

AVAILABLE DOMAIN EXPERTS:
{org_summary}
{session_note}{multihop_note}

USER QUESTION: {question}

Decompose into focused sub-questions. For each:
- Identify target domain expert(s).
- If a step depends on a previous answer, set depends_on to the 0-based step index.
- If the question references prior conversation, resolve the reference first.

Return JSON:
{{
    "sub_questions": [
        {{
            "question": "resolved sub-question text",
            "target_domains": ["domain_id_1"],
            "context": "Additional context for the expert",
            "depends_on": null
        }}
    ]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query decomposition and routing expert. Prefer the fewest domains needed. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "sub_questions": [{
                    "question": question,
                    "target_domains": [
                        d.domain_id for d in self.org_chart.domains[: self.max_routed_domains]
                    ],
                    "context": "",
                    "depends_on": None,
                }]
            }

        if isinstance(result, list):
            result = {"sub_questions": result}
        elif not isinstance(result, dict):
            result = {"sub_questions": []}
        sub_questions = result.get("sub_questions", [])
        if not isinstance(sub_questions, list):
            sub_questions = []
        normalized = []
        for sq in sub_questions:
            if isinstance(sq, str):
                sq = {"question": sq, "target_domains": [], "context": "", "depends_on": None}
            elif not isinstance(sq, dict):
                continue
            sq["target_domains"] = QAOrchestrator._normalize_target_domains(
                self, sq.get("target_domains", [])
            )
            sq.setdefault("depends_on", None)
            normalized.append(sq)
        if not normalized:
            normalized = [{
                "question": question,
                "target_domains": [
                    d.domain_id for d in self.org_chart.domains[: self.max_routed_domains]
                ],
                "context": "",
                "depends_on": None,
            }]
        result["sub_questions"] = normalized
        return result

    def _get_cross_domain_context(self, question: str) -> str:
        """Get cross-domain relations and multi-hop paths."""
        lines = []
        query_norm = normalize_entity_name(question)
        matched_entities = []
        for eid, entity in self.full_kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                n_norm = normalize_entity_name(n)
                if len(n_norm) > 2 and n_norm in query_norm:
                    matched_entities.append(eid)
                    break

        if self.org_chart.cross_domain_relations and matched_entities:
            relevant_cross = [
                t for t in self.org_chart.cross_domain_relations
                if t.subject in matched_entities or t.object in matched_entities
            ]
            if relevant_cross:
                lines.append("Cross-domain relationships:")
                for t in relevant_cross[:20]:
                    lines.append(f"  ({t.subject}) -[{t.relation}]-> ({t.object})")

        if len(matched_entities) >= 2:
            for i in range(len(matched_entities)):
                for j in range(i + 1, len(matched_entities)):
                    pths = find_paths(
                        self.full_kg,
                        matched_entities[i],
                        matched_entities[j],
                        max_hops=self.retrieval_config.max_hops,
                    )
                    if pths:
                        lines.append(f"\nPaths ({matched_entities[i]} → {matched_entities[j]}):")
                        lines.append(paths_to_text(pths))

        # GraphRAG-style global layer: thematic community summaries relevant
        # to the question (helps broad questions entity-local retrieval misses).
        if self.governed_kg is not None and getattr(self.governed_kg, "communities", None):
            from multi_agent_kg.core.community import community_context

            community_block = community_context(
                self.governed_kg,
                question,
                top_k=self.retrieval_config.community_top_k,
                vector_store=self.vector_store,
            )
            if community_block:
                lines.append(community_block)

        return "\n".join(lines) if lines else ""

    def _community_context(self, question: str) -> str:
        """Query-relevant community summaries (GraphRAG global layer).

        Borrowed QAOrchestrator methods (e.g. _build_global_fallback_context,
        called with an AdvancedQAOrchestrator as self) invoke
        self._community_context — this class must provide it or the whole QA
        query dies with AttributeError on the fallback path (latent since
        0932808; surfaced by multi-document ingestion triggering fallback
        more often — EXP-FRESHNESS-E2E local leg, 3/5 questions).
        """
        if self.governed_kg is None or not getattr(self.governed_kg, "communities", None):
            return ""
        from multi_agent_kg.core.community import community_context

        return community_context(
            self.governed_kg,
            question,
            top_k=self.retrieval_config.community_top_k,
            vector_store=self.vector_store,
        )

    def _format_debate_results(self, debate_results: List[Dict]) -> str:
        """Format debate resolutions for the synthesizer."""
        if not debate_results:
            return ""

        lines = ["RESOLVED CONFLICTS (from expert debate):"]
        for dr in debate_results:
            res = dr.get("resolution", {})
            lines.append(
                f"  - {dr['conflict'].get('description', '?')}: "
                f"RESOLVED → {res.get('resolution', '?')} "
                f"(confidence={res.get('confidence', 0):.2f})"
            )
        return "\n".join(lines)

    def _synthesize(
        self,
        question: str,
        domain_responses: List[Dict[str, Any]],
        cross_domain_context: str,
        debate_context: str,
    ) -> Dict[str, Any]:
        """Enhanced synthesis with debate resolutions."""
        if not domain_responses:
            return {
                "answer": "I don't have enough information in the knowledge graph to answer this question.",
                "coverage": 0.0,
                "confidence": 0.0,
                "gaps": ["No domain experts could provide relevant information"],
            }

        response_texts = []
        for resp in domain_responses:
            response_texts.append(
                f"Domain Expert [{resp.get('domain_id', '?')}] "
                f"(coverage={resp.get('coverage', 0):.2f}, "
                f"confidence={resp.get('confidence', 0):.2f}, "
                f"exploration_rounds={resp.get('exploration_rounds', 1)}):\n"
                f"  Answer: {resp.get('answer', 'N/A')}\n"
                f"  Evidence: {resp.get('evidence', [])}\n"
                f"  Out of scope: {resp.get('out_of_scope_aspects', [])}"
            )

        prompt = f"""You are a knowledge synthesis agent. Combine domain expert answers into
a single, coherent, well-cited response.

USER QUESTION: {question}

DOMAIN EXPERT RESPONSES:
{chr(10).join(response_texts)}

{f"CROSS-DOMAIN CONTEXT:{chr(10)}{cross_domain_context}" if cross_domain_context else ""}
{f"{chr(10)}{debate_context}" if debate_context else ""}
{build_rider(self.answer_format)}
RULES:
1. ONLY include claims that are supported by expert responses
2. If a conflict was resolved in debate, use the RESOLVED version
3. Prefer higher-confidence expert responses
4. Note any gaps (aspects no expert could answer)
5. Keep the final answer concise and avoid meta-commentary
6. Do NOT mention experts, routing, or internal system behavior in the answer text
7. Do NOT define entities or add background explanations unless explicitly supported by expert evidence
8. Limit the answer to at most {self.answer_format.max_sentences} sentences
9. For each major claim, include the supporting KG triple(s)
10. Provide TWO answer forms:
    * "answer": evidence-grounded prose response (1-{self.answer_format.max_sentences} sentences)
    * "short_answer": {self.answer_format.short_answer_instruction()} If evidence does not support an answer, set short_answer to "".

Examples of short_answer form:
- Q: "In which county is X located?" → short_answer: "Randall County"
- Q: "Who founded the company that distributed X?" → short_answer: "Mike Medavoy"
- Q: "Did the team win in 1990?" → short_answer: "yes"

Return JSON:
{{
    "answer": "Final answer prose here.",
    "short_answer": "minimal span",
    "coverage": 0.85,
    "confidence": 0.8,
    "gaps": ["unanswered aspects"],
    "key_claims": [
        {{
            "claim": "The specific claim",
            "source_domain": "domain_id",
            "evidence": "(entity) -[relation]-> (entity)"
        }}
    ]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledge synthesis expert. Combine expert answers "
                            "into a concise, KG-grounded response. "
                            "Avoid meta-commentary and return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "I do not have enough supported evidence to answer confidently.",
                "coverage": sum(r.get("coverage", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "confidence": sum(r.get("confidence", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "gaps": [],
                "key_claims": [],
            }

        if not isinstance(result, dict):
            result = {
                "answer": "I do not have enough supported evidence to answer confidently.",
                "coverage": sum(r.get("coverage", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "confidence": sum(r.get("confidence", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "gaps": [],
                "key_claims": [],
            }
        return result

    def _build_provenance(
        self, answer: str, domain_responses: List[Dict[str, Any]],
    ) -> ProvenanceChain:
        """Build provenance chain mapping answer claims to KG evidence."""
        provenance = ProvenanceChain()

        # Extract claims from the synthesized answer
        prompt = f"""Extract every factual claim from this answer. For each claim,
identify which entity/relationship it refers to.

ANSWER:
{answer}

Return JSON:
{{
    "claims": [
        {{
            "claim": "Metformin reduces HbA1c levels",
            "entities": ["metformin", "hba1c"],
            "relation": "reduces"
        }}
    ]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract factual claims. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            return provenance

        # Match claims to KG triples using aggressive normalization.
        # Triples store display names (e.g. "HOMA-IR") while entity IDs
        # use snake_case or stripped forms (e.g. "homair"), so we normalise
        # both sides before comparison.
        for claim_data in result.get("claims", []):
            claim_text = claim_data.get("claim", "")
            entities = claim_data.get("entities", [])

            # Resolve each mentioned entity to a set of normalised keys
            # that cover both entity IDs and display names in triples.
            resolved_norms: set = set()
            for eid_name in entities:
                agg_name = normalize_for_matching(eid_name)
                for eid, entity in self.full_kg.entities.items():
                    candidates = {normalize_for_matching(eid)}
                    for label in entity.labels:
                        candidates.add(normalize_for_matching(label))
                    if agg_name in candidates or any(
                        len(agg_name) >= 3 and (agg_name in c or c in agg_name)
                        for c in candidates
                    ):
                        resolved_norms.add(normalize_for_matching(eid))
                        for label in entity.labels:
                            resolved_norms.add(normalize_for_matching(label))
                        break

            # Find supporting triples — match using normalised forms
            supporting = []
            for t in self.full_kg.triples:
                subj_n = normalize_for_matching(t.subject)
                obj_n = normalize_for_matching(t.object)
                if subj_n in resolved_norms or obj_n in resolved_norms:
                    supporting.append(t)

            # For multi-entity claims, prefer triples that connect 2+ resolved entities
            if len(entities) >= 2 and supporting:
                connecting = [
                    t for t in supporting
                    if normalize_for_matching(t.subject) in resolved_norms
                    and normalize_for_matching(t.object) in resolved_norms
                ]
                if connecting:
                    supporting = connecting

            # Find source domain
            source_domain = ""
            for resp in domain_responses:
                resp_answer = resp.get("answer", "").lower()
                if any(e.lower() in resp_answer for e in entities):
                    source_domain = resp.get("domain_id", "")
                    break

            record = ProvenanceRecord(
                claim=claim_text,
                supporting_triples=supporting[:5],
                source_domain=source_domain,
                confidence=0.8 if supporting else 0.3,
                evidence_type="direct" if supporting else "ungrounded",
            )
            provenance.add(record)

        return provenance

    def _get_relevant_evidence(self, question: str, answer: str) -> str:
        """Gather KG evidence for critic review."""
        import re
        text_norm = normalize_entity_name(f"{question} {answer}")
        relevant = []

        for eid, entity in self.full_kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                if len(n) < 3:
                    continue
                n_norm = normalize_entity_name(n)
                if n_norm and n_norm in text_norm:
                    nbr = neighbourhood(self.full_kg, eid, hops=2)
                    relevant.extend(nbr)
                    break

        seen = set()
        lines = []
        for t in relevant:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                lines.append(f"({t.subject}) -[{t.relation}]-> ({t.object})")
        return "\n".join(lines[:80]) if lines else "No relevant KG evidence found."
