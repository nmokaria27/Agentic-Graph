"""jev_cascade retrieval mode, evidence rendering and the abstain gate (fake Jev, no network)."""
import pytest

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.governance import Domain
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core import retrievers
from multi_agent_kg.core.config import RetrievalConfig
from multi_agent_kg.core.qa_orchestrator import ABSTAIN_ANSWER, DomainExpertAgent, _format_triple


class FakeJev:
    def __init__(self, prob_for):
        self.prob_for, self.calls = prob_for, 0

    def ask(self, state, questions):
        self.calls += 1
        return {n: {"type": "noul", "p": self.prob_for(q["instructions"])} for n, q in questions.items()}


def _expert() -> DomainExpertAgent:
    kg = KnowledgeGraph()
    for eid in ("caroline", "support_group", "pottery", "melanie"):
        kg.add_entity(eid, [eid.replace("_", " ")], "THING")
    kg.add_triple("caroline", "ATTENDED", "support_group", confidence=0.9,
                  metadata={"evidence": "Caroline went to the LGBTQ support group on 7 May."})
    kg.add_triple("melanie", "LIKES", "pottery", confidence=0.9, metadata={"evidence": "Melanie loves pottery."})
    domain = Domain(domain_id="d1", label="D1", description="", entity_ids=set(kg.entities), relation_schema={})
    # No vector store -> the shortlist is the lexical lane only (no network).
    return DomainExpertAgent(domain=domain, full_kg=kg, llm_config=LLMConfig(model="test-model"),
                             retrieval_config=RetrievalConfig(retrieval_mode="jev_cascade"))


def test_format_triple_evidence_is_opt_in():
    expert = _expert()
    triple = expert.full_kg.get_active_triples()[0]
    assert _format_triple(triple) == "(caroline) -[ATTENDED]-> (support_group)"
    assert "7 May" in _format_triple(triple, with_evidence=True)


def test_jev_cascade_ranks_by_jev_and_records_gate(monkeypatch):
    fake = FakeJev(lambda text: 0.97 if "support group" in text else 0.02)
    monkeypatch.setattr(retrievers, "_JEV_CLIENT", fake)
    expert = _expert()
    triples, summary = expert._select_evidence("When did Caroline go to the support group?")
    assert summary is None and triples[0].object == "support_group"
    assert expert._last_gate["jev_max"] == pytest.approx(0.97) and fake.calls == 1


def test_jev_failure_falls_back_to_hybrid(monkeypatch):
    class Broken:
        def ask(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setattr(retrievers, "_JEV_CLIENT", Broken())
    expert = _expert()
    triples, _ = expert._select_evidence("Caroline support group")
    assert expert._last_gate is None and any(t.object == "support_group" for t in triples)


def test_abstain_gate_skips_llm_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(retrievers, "_JEV_CLIENT", FakeJev(lambda text: 0.05))
    monkeypatch.setenv("QA_ABSTAIN_GATE", "jev")
    expert = _expert()
    monkeypatch.setattr("multi_agent_kg.core.qa_orchestrator._chat_completion_json",
                        lambda *a, **k: pytest.fail("LLM must not be called when the gate abstains"))
    result = expert.answer("What is Caroline's favourite car?")
    assert result["abstained"] is True and result["answer"] == ABSTAIN_ANSWER


def test_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("QA_ABSTAIN_GATE", raising=False)
    monkeypatch.setattr(retrievers, "_JEV_CLIENT", FakeJev(lambda text: 0.05))
    expert = _expert()
    monkeypatch.setattr("multi_agent_kg.core.qa_orchestrator._chat_completion_json",
                        lambda *a, **k: {"answer": "x", "coverage": 0.1, "evidence": [], "confidence": 0.1})
    assert expert.answer("anything").get("abstained") is None


class IntentJev(FakeJev):
    def __init__(self, prob_for, intent):
        super().__init__(prob_for)
        self.intent = intent

    def ask(self, state, questions):
        if "intent" in questions:
            return {"intent": {"type": "choice", "choice": self.intent, "confidence": 0.9, "probabilities": {}}}
        return super().ask(state, questions)


def test_aggregate_intent_keeps_all_relevant_facts_and_exempts_gate(monkeypatch):
    monkeypatch.setenv("QA_INTENT_ROUTER", "jev")
    monkeypatch.setattr(retrievers, "_JEV_CLIENT", IntentJev(lambda text: 0.6, "aggregate"))
    expert = _expert()
    expert.retrieval_config.jev_top_k = 1
    triples, _ = expert._select_evidence("How many things do Caroline and Melanie like pottery support group?")
    assert len(triples) == 2  # top_k=1 is lifted for aggregates
    assert expert._last_gate["exempt"] is True and expert._last_intent == "aggregate"


def test_router_off_means_point_intent(monkeypatch):
    monkeypatch.delenv("QA_INTENT_ROUTER", raising=False)
    from multi_agent_kg.core.query_intent import classify_intent

    assert classify_intent("How many bikes do I own?", jev_client=None) == "point"
