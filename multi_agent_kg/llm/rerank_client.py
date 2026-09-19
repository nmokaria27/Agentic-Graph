"""
Pointwise reranker client (Fireworks ``/rerank``, e.g. qwen3-reranker-8b).

Scores are per (query, document) pair, so a large document list is split into
chunks that run concurrently and the scores stay comparable across chunks.
Results are disk-cached by (model, query, document).

Env: ``FIREWORKS_API_KEY``, ``FIREWORKS_BASE_URL``, ``RERANK_MODEL``, ``RERANK_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import httpx

DEFAULT_RERANK_MODEL = "accounts/fireworks/models/qwen3-reranker-8b"


class RerankClient:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        chunk_size: int = 100,
        max_workers: int = 8,
        max_retries: int = 5,
    ) -> None:
        self.model = model or os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self.base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")
        cache = cache_dir or os.getenv("RERANK_CACHE_DIR")
        self.cache_path = Path(cache) / "rerank_cache.json" if cache else None
        self._cache: Dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.chunk_size = max(1, chunk_size)
        self.max_workers = max(1, max_workers)
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "tokens": 0, "seconds": 0.0}

    def _key(self, query: str, document: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{query}\x00{document}".encode("utf-8")).hexdigest()

    def _post(self, query: str, documents: List[str]) -> List[float]:
        headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
        payload = {"model": self.model, "query": query, "documents": documents, "return_documents": False}
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            response = httpx.post(f"{self.base_url}/rerank", headers=headers, json=payload, timeout=120)
            if response.status_code in (429, 500, 502, 503, 529) and attempt < self.max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            response.raise_for_status()
            body = response.json()
            with self._lock:
                self.stats["calls"] += 1
                self.stats["tokens"] += int((body.get("usage") or {}).get("total_tokens", 0))
                self.stats["seconds"] += time.perf_counter() - started
            scores = [0.0] * len(documents)
            for item in body["data"]:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        raise RuntimeError("unreachable")

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Relevance score per document, in input order."""
        keys = [self._key(query, d) for d in documents]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        with self._lock:
            self.stats["cache_hits"] += len(documents) - len(missing)
        chunks = [missing[i : i + self.chunk_size] for i in range(0, len(missing), self.chunk_size)]
        if chunks:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
                results = list(pool.map(lambda idx: self._post(query, [documents[i] for i in idx]), chunks))
            with self._lock:
                for idx, scores in zip(chunks, results):
                    for i, s in zip(idx, scores):
                        self._cache[keys[i]] = s
        return [self._cache[k] for k in keys]

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            os.replace(tmp, self.cache_path)
