"""GB-2d query-time value collisions: shape pairing, Jev same-slot decision, date ordering in code."""
from multi_agent_kg.core.knowledge_graph import Triple
from multi_agent_kg.core.value_collisions import candidate_pairs, mark_stale_values, shape_of


def _t(rel, obj, date):
    return Triple("user", rel, obj, 0.9, metadata={"provenance": {"refs": [{"document_date": date}]}})


TRIPLES = [
    _t("PRE-APPROVED_FOR_LOAN_AMOUNT", "$350,000", "2023/05/01"),
    _t("HAS_PRE_APPROVAL_AMOUNT", "$400,000", "2023/06/10"),
    _t("PAYS_RENT", "$1,200", "2023/05/01"),
    _t("LIVES_IN", "the suburbs", "2023/05/01"),
]


class FakeJev:
    def ask_many(self, items):
        out = []
        for state, _ in items:
            same = "350,000" in state["fact_a"] + state["fact_b"] and "400,000" in state["fact_a"] + state["fact_b"]
            out.append({"same": {"type": "score", "score": 2.0 if same else 0.0, "confidence": 0.9,
                                 "probabilities": {"0": 0.05 if same else 0.95, "1": 0.0, "2": 0.95 if same else 0.05}}})
        return out


def test_shapes_and_pairs():
    assert shape_of("$350,000") == "money" and shape_of("25:50") == "clock"
    assert shape_of("three times a week") == "frequency" and shape_of("the suburbs") is None
    pairs = candidate_pairs(TRIPLES, lambda t: t.object)
    assert (0, 1) in pairs and all(3 not in pair for pair in pairs)


def test_older_same_slot_value_is_marked_on_a_copy():
    marked, decisions = mark_stale_values(TRIPLES, lambda t: t.object, lambda t: f"{t.relation} {t.object}", FakeJev())
    assert marked[0].metadata.get("superseded_by") and not marked[1].metadata.get("superseded_by")
    assert not marked[2].metadata.get("superseded_by")  # rent is a different slot
    assert "superseded_by" not in TRIPLES[0].metadata  # original untouched
    assert any(d["stale"] == "a" for d in decisions)
