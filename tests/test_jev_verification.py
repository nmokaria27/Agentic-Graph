"""VERIFY_BACKEND=jev: triples are gated by Jev P(supported), LLM is the fallback."""
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent

TRIPLES = [
    {"subject": "CRF", "relation": "Used-for", "object": "NER", "confidence": 0.9},
    {"subject": "CRF", "relation": "Part-of", "object": "parser", "confidence": 0.9},
    {"subject": "it", "relation": "Used-for", "object": "NER", "confidence": 0.9},
]


class FakeJev:
    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        return {name: {"type": "noul", "p": p} for name, p in zip(questions, self.probs)}


def _agent(monkeypatch, fake):
    monkeypatch.setenv("VERIFY_BACKEND", "jev")
    agent = ExtractionVerificationAgent()
    agent._jev_client = fake
    return agent


def test_jev_maps_probability_to_status_in_one_call(monkeypatch):
    fake = FakeJev([0.9, 0.4, 0.05])
    result = _agent(monkeypatch, fake)._verify_with_jev("We use a CRF for NER.", TRIPLES)
    statuses = [v["verification_status"] for v in result["verified_triples"]]
    assert statuses == ["verified", "partial", "rejected"]
    assert len(fake.calls) == 1 and len(fake.calls[0][1]) == 3
    assert result["verified_triples"][0]["final_confidence"] == 0.9
    assert result["verification_summary"] == {
        "total": 3, "verified": 1, "partial": 1, "rejected": 1, "hallucinated": 0}


def test_jev_failure_falls_back_to_llm(monkeypatch):
    class Broken:
        def ask(self, *a, **k):
            raise RuntimeError("429")

    agent = _agent(monkeypatch, Broken())
    monkeypatch.setattr(agent, "_verify_against_source", lambda text, triples: {"verified_triples": ["llm"]})
    assert agent._verify_with_jev("text", TRIPLES) == {"verified_triples": ["llm"]}


def test_default_backend_is_llm(monkeypatch):
    monkeypatch.delenv("VERIFY_BACKEND", raising=False)
    assert ExtractionVerificationAgent().verify_backend == "llm"
