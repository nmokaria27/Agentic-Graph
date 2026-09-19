"""
Thin wrapper around TypeSafe's Jev (System One) decision model.

Jev does not generate text: it answers typed questions about a ``state`` —
``noul`` (P(yes)), ``choice`` (pick one key from a closed set) and ``score``
(ordered rubric level). This module keeps those calls cheap to experiment with:

- questions are plain dicts (hashable → disk cache keyed on state+questions),
- answers come back as plain dicts, independent of SDK response classes,
- large question sets are split across calls and run on a thread pool,
- per-call usage/latency is appended to a JSONL log (``JEV_USAGE_LOG``).

Question dict shapes::

    {"type": "noul",   "instructions": "..."}
    {"type": "choice", "instructions": "...", "criteria": {"key": "description", ...}}
    {"type": "score",  "instructions": "...", "criteria": ["lowest level", ..., "highest"]}

Answer dict shapes::

    {"type": "noul",   "p": 0.93}
    {"type": "choice", "choice": "key", "confidence": 0.8, "probabilities": {...}}
    {"type": "score",  "score": 1.7, "confidence": 0.5, "probabilities": {"0": .., "1": ..}}

Env: ``TYPESAFE_API_KEY`` (required for live calls), ``TYPESAFE_DEFAULT_MODEL``,
``JEV_CACHE_DIR``, ``JEV_USAGE_LOG``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Jev input price (USD per input token); output tokens are free.
JEV_INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000

Questions = Mapping[str, Mapping[str, Any]]
Answers = Dict[str, Dict[str, Any]]


def _to_sdk_question(spec: Mapping[str, Any]) -> Any:
    from typesafe_sdk import Choice, Noul, Score

    qtype = spec["type"]
    instructions = spec.get("instructions")
    if qtype == "noul":
        return Noul(instructions=instructions)
    if qtype == "choice":
        return Choice(instructions=instructions, criteria=dict(spec["criteria"]))
    if qtype == "score":
        return Score(instructions=instructions, criteria=list(spec["criteria"]))
    raise ValueError(f"unknown Jev question type: {qtype!r}")


def _answer_to_dict(answer: Any) -> Dict[str, Any]:
    # Score answers also carry `probabilities`/`confidence`; check the
    # discriminating attribute first.
    if hasattr(answer, "noul"):
        return {"type": "noul", "p": float(answer.noul)}
    if hasattr(answer, "score"):
        return {
            "type": "score",
            "score": float(answer.score),
            "confidence": float(answer.confidence),
            "probabilities": {str(k): float(v) for k, v in dict(answer.probabilities).items()},
        }
    if hasattr(answer, "choice"):
        return {
            "type": "choice",
            "choice": answer.choice,
            "confidence": float(answer.confidence),
            "probabilities": {str(k): float(v) for k, v in dict(answer.probabilities).items()},
        }
    raise ValueError(f"unrecognised Jev answer: {answer!r}")


def _cache_key(model: str, state: Any, questions: Questions) -> str:
    blob = json.dumps({"m": model, "s": state, "q": questions}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JevClient:
    """Cached, batched access to Jev. Thread-safe."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        usage_log: Optional[str] = None,
        max_questions_per_call: int = 32,
        max_workers: int = 8,
        max_inflight: int = 4,
        max_retries: int = 6,
        sdk_client: Any = None,
    ) -> None:
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        cache = cache_dir or os.getenv("JEV_CACHE_DIR")
        self.cache_dir = Path(cache) if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log = usage_log or os.getenv("JEV_USAGE_LOG")
        self.max_questions_per_call = max(1, max_questions_per_call)
        self.max_workers = max(1, max_workers)
        self._sdk = sdk_client
        self.max_retries = max(0, max_retries)
        # Caps concurrent HTTP calls across nested ask()/ask_many() pools.
        self._inflight = threading.BoundedSemaphore(max(1, max_inflight))
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "seconds": 0.0, "retries": 0}

    # ── internals ────────────────────────────────────────────────────
    def _client(self) -> Any:
        if self._sdk is None:
            from typesafe_sdk import TypeSafeClient

            self._sdk = TypeSafeClient(model=self.model, timeout=60.0)
        return self._sdk

    def _cache_get(self, key: str) -> Optional[Answers]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, answers: Answers) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(answers), encoding="utf-8")
        os.replace(tmp, path)

    def _log(self, record: Dict[str, Any]) -> None:
        if not self.usage_log:
            return
        with self._lock:
            with open(self.usage_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def _call(self, state: Any, questions: Questions) -> Answers:
        key = _cache_key(self.model, state, questions)
        cached = self._cache_get(key)
        if cached is not None:
            with self._lock:
                self.stats["cache_hits"] += 1
            return cached
        sdk_questions = {name: _to_sdk_question(spec) for name, spec in questions.items()}
        response, elapsed = self._send(state, sdk_questions)
        answers = {name: _answer_to_dict(ans) for name, ans in response.answers.items()}
        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        with self._lock:
            self.stats["calls"] += 1
            self.stats["input_tokens"] += input_tokens
            self.stats["seconds"] += elapsed
        self._log(
            {
                "ts": time.time(),
                "model": getattr(response, "model", self.model),
                "questions": len(questions),
                "input_tokens": input_tokens,
                "seconds": round(elapsed, 4),
            }
        )
        self._cache_put(key, answers)
        return answers

    def _send(self, state: Any, sdk_questions: Mapping[str, Any]) -> Tuple[Any, float]:
        """One HTTP call with backoff on 429 (rate limit) / 529 (overloaded)."""
        from typesafe_sdk import TypeSafeAPIError, TypeSafeRateLimitError

        for attempt in range(self.max_retries + 1):
            with self._inflight:
                started = time.perf_counter()
                try:
                    response = self._client().system_one(state=state, questions=sdk_questions)
                    return response, time.perf_counter() - started
                except TypeSafeAPIError as exc:
                    retryable = isinstance(exc, TypeSafeRateLimitError) or getattr(exc, "status", None) == 529
                    if not retryable or attempt == self.max_retries:
                        raise
            with self._lock:
                self.stats["retries"] += 1
            time.sleep(min(30.0, 1.0 * 2 ** attempt))
        raise RuntimeError("unreachable")

    # ── public API ───────────────────────────────────────────────────
    def ask(self, state: Any, questions: Questions) -> Answers:
        """Ask all ``questions`` about one ``state``; splits into ≤N-question calls."""
        names = list(questions)
        chunks = [
            {n: questions[n] for n in names[i : i + self.max_questions_per_call]}
            for i in range(0, len(names), self.max_questions_per_call)
        ]
        if len(chunks) == 1:
            return self._call(state, chunks[0])
        merged: Answers = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
            for part in pool.map(lambda c: self._call(state, c), chunks):
                merged.update(part)
        return merged

    def ask_many(self, items: Sequence[Tuple[Any, Questions]]) -> List[Answers]:
        """Run many independent (state, questions) requests concurrently, order kept."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda item: self.ask(item[0], item[1]), items))

    def cost_usd(self) -> float:
        return self.stats["input_tokens"] * JEV_INPUT_PRICE_PER_TOKEN
