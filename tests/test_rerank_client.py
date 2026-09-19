"""RerankClient: chunked scoring keeps input order; cache avoids repeat calls."""
from multi_agent_kg.llm.rerank_client import RerankClient


def test_scores_keep_order_across_chunks_and_cache(tmp_path, monkeypatch):
    client = RerankClient(cache_dir=str(tmp_path), chunk_size=2)
    calls = []

    def fake_post(query, documents):
        calls.append(list(documents))
        return [float(len(d)) for d in documents]

    monkeypatch.setattr(client, "_post", fake_post)
    docs = ["a", "bbb", "cc", "dddd", "e"]
    assert client.score("q", docs) == [1.0, 3.0, 2.0, 4.0, 1.0]
    assert sorted(len(c) for c in calls) == [1, 2, 2]
    client.save_cache()

    again = RerankClient(cache_dir=str(tmp_path), chunk_size=2)
    monkeypatch.setattr(again, "_post", lambda q, d: (_ for _ in ()).throw(AssertionError("no call expected")))
    assert again.score("q", docs) == [1.0, 3.0, 2.0, 4.0, 1.0]
