"""JevClient: question/answer translation, chunking, caching, usage accounting.

Uses a fake SDK client so no network or API key is needed.
"""
import json
from types import SimpleNamespace

from multi_agent_kg.llm.typesafe_client import JEV_INPUT_PRICE_PER_TOKEN, JevClient


class FakeSDK:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, dict(questions)))
        answers = {}
        for name, q in questions.items():
            kind = type(q).__name__
            if kind == "Noul":
                answers[name] = SimpleNamespace(noul=0.9)
            elif kind == "Choice":
                answers[name] = SimpleNamespace(choice="b", confidence=0.7, probabilities={"a": 0.3, "b": 0.7})
            else:
                answers[name] = SimpleNamespace(score=1.5, confidence=0.4, probabilities={0: 0.2, 1: 0.3, 2: 0.5})
        return SimpleNamespace(model="jev-test", answers=answers, usage=SimpleNamespace(input_tokens=100))


QUESTIONS = {
    "yes": {"type": "noul", "instructions": "Is it true?"},
    "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}},
    "rate": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]},
}


def test_answers_are_plain_dicts_per_type():
    client = JevClient(sdk_client=FakeSDK())
    answers = client.ask("state", QUESTIONS)
    assert answers["yes"] == {"type": "noul", "p": 0.9}
    assert answers["pick"]["type"] == "choice" and answers["pick"]["choice"] == "b"
    assert answers["rate"]["type"] == "score" and answers["rate"]["score"] == 1.5
    assert answers["rate"]["probabilities"] == {"0": 0.2, "1": 0.3, "2": 0.5}
    json.dumps(answers)  # cacheable as JSON


def test_large_question_sets_are_chunked_and_merged():
    sdk = FakeSDK()
    client = JevClient(sdk_client=sdk, max_questions_per_call=2)
    questions = {f"q{i}": {"type": "noul", "instructions": f"q{i}?"} for i in range(5)}
    answers = client.ask({"text": "x"}, questions)
    assert set(answers) == set(questions)
    assert sorted(len(q) for _, q in sdk.calls) == [1, 2, 2]
    assert client.stats["input_tokens"] == 300


def test_disk_cache_skips_repeat_calls(tmp_path):
    sdk = FakeSDK()
    client = JevClient(sdk_client=sdk, cache_dir=str(tmp_path))
    first = client.ask("s", QUESTIONS)
    second = JevClient(sdk_client=sdk, cache_dir=str(tmp_path)).ask("s", QUESTIONS)
    assert first == second
    assert len(sdk.calls) == 1


def test_ask_many_preserves_order_and_logs_usage(tmp_path):
    log = tmp_path / "usage.jsonl"
    client = JevClient(sdk_client=FakeSDK(), usage_log=str(log), max_workers=4)
    items = [(f"state-{i}", {"yes": QUESTIONS["yes"]}) for i in range(6)]
    results = client.ask_many(items)
    assert len(results) == 6 and all(r["yes"]["p"] == 0.9 for r in results)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 6 and records[0]["input_tokens"] == 100
    assert client.cost_usd() == 600 * JEV_INPUT_PRICE_PER_TOKEN
