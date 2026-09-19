"""
Configuration classes for the multi-agent knowledge graph system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RelationType:
    """
    Defines a type of relation that can exist in the knowledge graph.

    Attributes:
        name: The name/label of the relation (e.g., "treats", "causes")
        description: A clear description of what this relation means
        allowed_subject_types: Optional list of entity types that can be subjects
        allowed_object_types: Optional list of entity types that can be objects
    """

    name: str
    description: str
    allowed_subject_types: Optional[List[str]] = None
    allowed_object_types: Optional[List[str]] = None

    def __repr__(self) -> str:
        return f"RelationType({self.name})"

    def validate_triple(self, subject_type: Optional[str], object_type: Optional[str]) -> bool:
        """
        Check if a triple with given subject/object types is valid for this relation.

        Args:
            subject_type: Type of the subject entity
            object_type: Type of the object entity

        Returns:
            True if the triple is valid, False otherwise
        """
        if self.allowed_subject_types and subject_type:
            if subject_type not in self.allowed_subject_types:
                return False

        if self.allowed_object_types and object_type:
            if object_type not in self.allowed_object_types:
                return False

        return True


@dataclass
class RelationSchema:
    """
    Collection of relation types that define the schema for the knowledge graph.

    Attributes:
        types: List of allowed relation types
        allow_new_types: Whether to allow discovery of new relation types
    """

    types: List[RelationType] = field(default_factory=list)
    allow_new_types: bool = False

    def get_relation_type(self, name: str) -> Optional[RelationType]:
        """Get a relation type by name."""
        for rel_type in self.types:
            if rel_type.name == name:
                return rel_type
        return None

    def has_relation_type(self, name: str) -> bool:
        """Check if a relation type exists in the schema."""
        return self.get_relation_type(name) is not None

    def add_relation_type(self, relation_type: RelationType) -> None:
        """Add a new relation type to the schema."""
        if not self.has_relation_type(relation_type.name):
            self.types.append(relation_type)

    def get_schema_description(self) -> str:
        """Get a formatted description of all relation types for LLM prompts."""
        lines = ["Available relation types:"]
        for rel_type in self.types:
            lines.append(f"  - {rel_type.name}: {rel_type.description}")
            if rel_type.allowed_subject_types:
                lines.append(f"    Subject types: {', '.join(rel_type.allowed_subject_types)}")
            if rel_type.allowed_object_types:
                lines.append(f"    Object types: {', '.join(rel_type.allowed_object_types)}")
        return "\n".join(lines)


@dataclass
class LLMConfig:
    """
    Configuration for LLM calls.

    Attributes:
        model: The model to use (e.g., "gemma3:27b", "qwen3:8b")
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
    """

    model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b"))
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API calls."""
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens:
            config["max_tokens"] = self.max_tokens
        return config


@dataclass
class AnswerFormatConfig:
    """
    Controls answer verbosity / short-answer span shape during QA synthesis.

    This is deliberately GENERIC — it carries no benchmark identity. Named,
    benchmark-specific profiles (e.g. terse-for-LoComo, exact-span-for-MAB) live in
    the evaluation layer (evaluation/answer_format_profiles.py) and are injected as an
    AnswerFormatConfig instance. Core QA code only ever consumes this object, never a
    benchmark name. Defaults reproduce the historical prompt wording.

    Attributes:
        max_sentences: cap on the prose answer length.
        short_answer_words: cap on the minimal short-answer span.
        style: shape of the short answer — "minimal_span" (name/place/date only),
            "short_dry" (terse phrase), or "verbose" (full sentence allowed).
        commit_mode: when True, inject the anti-hedge rider so the model commits to a
            best-effort span instead of refusing. Defaults to the KGQA_COMMIT_MODE env.
    """

    max_sentences: int = 3
    short_answer_words: int = 5
    style: str = "minimal_span"
    commit_mode: bool = field(
        default_factory=lambda: os.getenv("KGQA_COMMIT_MODE") == "1"
    )

    _VALID_STYLES = {"minimal_span", "short_dry", "verbose"}

    def __post_init__(self) -> None:
        if self.style not in self._VALID_STYLES:
            raise ValueError(
                f"Unknown answer style {self.style!r}; expected one of "
                f"{sorted(self._VALID_STYLES)}"
            )

    def short_answer_instruction(self) -> str:
        """Style-specific wording for the short_answer span, sized by short_answer_words.

        Consumed by the QA synthesis prompts so `style` actually changes prompt shape
        (minimal_span ≠ short_dry ≠ verbose), instead of being an inert field.
        """
        words = self.short_answer_words
        if self.style == "minimal_span":
            return (
                f"the MINIMAL span (1-{words} words) that directly answers the question. "
                "For a person, the name only. For a place, the place name only. For a "
                'date, the date only. For yes/no questions, "yes" or "no".'
            )
        if self.style == "short_dry":
            return (
                f"a short, dry phrase (1-{words} words) — terse, no full sentence and "
                'no surrounding words. For yes/no questions, "yes" or "no".'
            )
        # verbose
        return (
            f"a concise answer (up to {words} words); a short sentence is acceptable if "
            'it stays under the word limit. For yes/no questions, "yes" or "no".'
        )


@dataclass
class RetrievalConfig:
    """
    Configuration for hybrid (graph + vector) retrieval.

    Modes:
        lexical: legacy exact/regex matching only (ablation baseline)
        dense:   vector search only (ablation)
        hybrid:  lexical ∪ vector fused with RRF, then graph expansion (default)
    """

    retrieval_mode: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    )
    embedding_model: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL") or None
    )
    entity_top_k: int = 10
    triple_top_k: int = 40
    domain_top_k: int = 4
    entity_min_score: float = 0.62
    resolution_min_score: float = 0.75
    relation_schema_min_score: float = 0.85

    # Retrieval shaping knobs (promoted from hardcoded literals so they can be tuned).
    # Defaults reproduce the previous in-code constants exactly — zero behaviour change.
    seed_cap: int = 40  # max seed triples before graph expansion
    lexical_seed_k: int = 40  # top lexical-scored triples kept as seeds
    focused_limit: int = 60  # final query-focused triples returned to the prompt
    max_hops: int = 3  # BFS depth for multi-hop path finding
    neighbourhood_hops: int = 3  # hops for single-entity neighbourhood expansion
    neighbourhood_display: int = 50  # max neighbourhood triples rendered into context
    summary_trigger: int = 40  # subgraph size above which graph_summary mode summarizes

    # Personalized-PageRank expansion: entities ranked by PPR from the seed
    # set join the expansion frontier and add a scoring boost. QA_USE_PPR=0
    # disables (falls back to plain 1-hop seed adjacency).
    use_ppr: bool = field(
        default_factory=lambda: os.getenv("QA_USE_PPR", "1") != "0"
    )
    ppr_top_k: int = 25  # entities kept from the PPR ranking
    ppr_boost: float = 3.0  # scoring weight of normalized PPR mass per triple

    # Community summaries (GraphRAG-style global layer) added to QA context.
    community_top_k: int = 3

    # graph_completion retriever (Phase 2): vector-seeded k-hop expansion knobs.
    # Only consulted when retrieval_mode == "graph_completion"; defaults leave every
    # other mode's behaviour untouched.
    neighbourhood_seed_top_k: int = 10  # max vector-seed entities expanded to k-hop
    seed_weight_triple: float = 1.0  # fusion weight for triple-collection seeds
    seed_weight_entity: float = 1.0  # fusion weight for entity-collection seeds

    # jev_cascade (EXP-JEV-3): dense top-``jev_pool`` ∪ lexical top-``jev_lexical_k``
    # → one Jev Noul per fact → top-``jev_top_k``. ``jev_abstain_tau`` is the
    # abstain-gate threshold on the best per-fact relevance (QA_ABSTAIN_GATE=jev).
    jev_pool: int = 200
    jev_lexical_k: int = 20
    jev_top_k: int = 15
    jev_abstain_tau: float = field(
        default_factory=lambda: float(os.getenv("QA_JEV_ABSTAIN_TAU", "0.93"))
    )

    # Retrieval strategies that branch the evidence-assembly path. lexical/dense/hybrid
    # are the historical fused modes; graph_completion/graph_summary/chunk are explicit
    # retriever switches (see DomainExpertAgent._select_evidence).
    _VALID_MODES = {
        "lexical",
        "dense",
        "hybrid",
        "graph_completion",
        "graph_summary",
        "chunk",
        "jev_cascade",
    }

    def __post_init__(self) -> None:
        self.retrieval_mode = (self.retrieval_mode or "hybrid").lower()
        if self.retrieval_mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown retrieval_mode {self.retrieval_mode!r}; "
                f"expected one of {sorted(self._VALID_MODES)}"
            )

    @property
    def use_vectors(self) -> bool:
        # All modes except pure lexical rely on the vector index.
        return self.retrieval_mode != "lexical"

    @property
    def use_lexical(self) -> bool:
        # Pure dense skips lexical seeding; every other mode keeps a hybrid lexical base.
        return self.retrieval_mode != "dense"
