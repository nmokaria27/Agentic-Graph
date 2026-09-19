"""
LLM client wrapper supporting Ollama (default), OpenAI, and VLLM backends.

Backend selection:
- Set LLM_BACKEND=openai  to use OpenAI (requires OPENAI_API_KEY)
- Set LLM_BACKEND=vllm    to use a VLLM server (set VLLM_BASE_URL, e.g. http://gpu-host:8000/v1)
- Default: Ollama at http://localhost:11434 (via SSH tunnel to GPU)
"""

from typing import List, Dict, Any, Optional, Tuple, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import os
import json
import re
import time
from dotenv import load_dotenv

T = TypeVar("T", bound=BaseModel)

load_dotenv()

# ── Backend configuration ─────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# Default chat model for callsites that don't pass one explicitly. Follows
# LLM_DEFAULT_MODEL so vLLM/OpenAI deployments aren't hit with the Ollama name.
DEFAULT_CHAT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")

# Timeout for local/remote model calls
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # 5 min default
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "3.0"))
_RETRY_BACKOFF_CAP = float(os.getenv("LLM_RETRY_BACKOFF_CAP", "60.0"))
# When Ollama/VLLM is unreachable (tunnel dropped, process restart), keep polling
# the health endpoint for this many seconds before giving up on the call.
_HEALTHCHECK_TIMEOUT = float(os.getenv("LLM_HEALTHCHECK_TIMEOUT", "300"))

if LLM_BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_BACKEND == "vllm":
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout=_TIMEOUT,
    )
else:
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=_TIMEOUT,
    )

# ── Fireworks (hosted) chat routing ───────────────────────────────────
# Models named "accounts/fireworks/models/..." (or the "fireworks/<name>"
# shorthand) route to the Fireworks serverless API regardless of LLM_BACKEND,
# so a request can pick a hosted model while the default stays on local vLLM.
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
_fireworks_client: Optional[OpenAI] = None


def _is_fireworks_model(model: str) -> bool:
    return model.startswith("accounts/fireworks/") or model.startswith("fireworks/")


def _client_for_model(resolved_model: str) -> OpenAI:
    global _fireworks_client
    if _is_fireworks_model(resolved_model):
        key = os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise Exception(
                f"FIREWORKS_API_KEY is not set — cannot route {resolved_model!r} to Fireworks"
            )
        if _fireworks_client is None:
            _fireworks_client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=key, timeout=_TIMEOUT)
        return _fireworks_client
    return client


# ── Embedding backend (may differ from chat) ──────────────────────────
# Embeddings can target a separate OpenAI-compatible server than chat. Common
# setup: chat on vLLM (gemma), embeddings on Ollama's mxbai-embed-large via the
# existing tunnel. Set EMBEDDING_BASE_URL to route embedding calls elsewhere;
# otherwise the main chat client is reused.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
if EMBEDDING_BASE_URL:
    embed_client = OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=os.getenv("EMBEDDING_API_KEY", "ollama"),
        timeout=_TIMEOUT,
    )
else:
    embed_client = client

# Optional per-call usage logging. When LLM_USAGE_LOG points to a file path,
# each completion appends one JSON line with {model, prompt_tokens,
# completion_tokens, cached_tokens, reasoning_tokens}. Used for cost analysis.
_USAGE_LOG_PATH = os.getenv("LLM_USAGE_LOG")


def _log_usage(model: str, response: Any) -> None:
    if not _USAGE_LOG_PATH:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    record = {
        "ts": time.time(),
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cached_tokens": getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0,
    }
    try:
        with open(_USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _backend_is_up() -> bool:
    """Quick TCP probe of the active backend health endpoint. Returns False on any error."""
    if LLM_BACKEND == "openai":
        return True
    try:
        import urllib.request
        if LLM_BACKEND == "vllm":
            # VLLM exposes /health on its HTTP server
            base = VLLM_BASE_URL.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            url = f"{base}/health"
        else:
            base = OLLAMA_BASE_URL.rstrip("/")
            # OLLAMA_BASE_URL commonly includes /v1; strip it for the native health endpoint.
            if base.endswith("/v1"):
                base = base[:-3]
            url = f"{base}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


# Keep old name as alias so any external callers are not broken
_ollama_is_up = _backend_is_up


def _wait_for_ollama(deadline: float) -> bool:
    """Poll the active backend until it responds or the deadline passes. Returns True on recovery."""
    while time.time() < deadline:
        if _backend_is_up():
            return True
        time.sleep(5.0)
    return False

# Map old OpenAI model names → Ollama equivalents (for backward compat)
_OPENAI_TO_OLLAMA = {
    "gpt-4o": "gemma4:31b",
    "gpt-4o-mini": "gemma4:31b",
    "gpt-4": "gemma4:31b",
    "gpt-4-turbo": "gemma4:31b",
    "gpt-3.5-turbo": "gemma4:31b",
}


def _resolve_model(model: str) -> str:
    """Resolve model name: translate OpenAI names to Ollama/VLLM when using those backends."""
    if model.startswith("fireworks/"):
        return "accounts/fireworks/models/" + model[len("fireworks/"):]
    if _is_fireworks_model(model) or LLM_BACKEND in ("openai", "vllm"):
        return model
    return _OPENAI_TO_OLLAMA.get(model, model)


# Model families known to emit <think>...</think> reasoning tokens.
# On Ollama: GBNF json_object grammar blocks the leading '<' → empty content.
# On VLLM:  json_object mode strips special tokens; thinking models may still
#            embed reasoning tokens in the visible output, so we skip constrained
#            decoding for them and rely on our robust _extract_json parser.
_THINKING_MODEL_PATTERNS = (
    "gemma4",       # Google Gemma 4 family (gemma4:12b, gemma4:27b, gemma4:31b)
    "gemma-4",      # vLLM HuggingFace naming (google/gemma-4-31B-it)
    "deepseek-r1",  # DeepSeek R1 family
    "deepseek-v4",  # DeepSeek V4 family (v4-pro, v4-flash) — CoT prose before JSON;
                    # disable json_object mode and let _extract_json dig out the JSON
    "qwen3",        # Qwen 3.x (qwen3, qwen3p6/3p7-plus) — thinking by default
    "qwq",          # Qwen QwQ reasoning models
    "gpt-oss",      # OpenAI gpt-oss-120b/20b — harmony reasoning channel
    "minimax-m",    # MiniMax M2.x / M3 — reasoning models (verbose CoT output)
    "nemotron",     # NVIDIA Nemotron (3 Ultra etc.) — reasoning-capable
    "llama-3.3",    # Llama 3.3 instruct (sometimes emits chain-of-thought preamble)
    # NOTE: GLM-5 (glm-5p2) and Kimi K2 are intentionally NOT here — GLM json_object
    # mode is verified working in this project, and Kimi K2 is a non-thinking
    # instruct model. Add "glm-5"/"kimi-k2-thinking" only if a switch starts failing.
)


# NVIDIA Nemotron exposes an explicit reasoning switch via a system directive
# ("detailed thinking on" / "detailed thinking off"). Reasoning OFF is ~4x faster, BUT the
# current verbose multi-stage RHF prompts make Nemotron reason past a small budget anyway
# and truncate to EMPTY output (finish=length) when reasoning is forced off — so default is
# ON to preserve correctness. The switch stays available for experiments: with SHORT
# single-pass prompts, NEMOTRON_THINKING=off is fast AND complete. See EXTRACTION_EXPERIMENTS.md
NEMOTRON_THINKING = os.getenv("NEMOTRON_THINKING", "on").strip().lower()


def _is_nemotron(model_name: str) -> bool:
    return "nemotron" in model_name.lower()


def _nemotron_reasoning_suppressed(model_name: str) -> bool:
    """True when this is a Nemotron model AND we want reasoning turned off."""
    return _is_nemotron(model_name) and NEMOTRON_THINKING != "on"


def _model_is_thinking(model_name: str) -> bool:
    """Return True if the model is known to emit reasoning/thinking tokens.

    A Nemotron model with reasoning suppressed behaves like a normal instruct model
    (no <think> blocks), so it flows through the standard JSON-mode / non-inflated path.
    """
    if _nemotron_reasoning_suppressed(model_name):
        return False
    name_lower = model_name.lower()
    return any(pat in name_lower for pat in _THINKING_MODEL_PATTERNS)


def _apply_nemotron_reasoning_directive(
    messages: List[Dict[str, str]], resolved_model: str
) -> List[Dict[str, str]]:
    """Prepend/merge Nemotron's 'detailed thinking off' control into the system msg."""
    if not _nemotron_reasoning_suppressed(resolved_model):
        return messages
    directive = "detailed thinking off"
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            if "detailed thinking" not in m["content"].lower():
                m["content"] = f"{directive}\n{m['content']}"
            return out
    return [{"role": "system", "content": directive}, *out]


def _extract_json(text: str) -> Any:
    """
    Robustly extract and parse JSON from LLM output that may contain
    markdown fences, preamble text, thinking tags, or trailing commentary.

    Handles thinking-model output where:
    - <think>...</think> blocks precede the JSON (possibly unclosed if truncated)
    - JSON may be truncated mid-value due to token limits
    - Markdown fences may or may not be present
    """
    if not text or not text.strip():
        return None

    # 1. Strip <think>...</think> blocks (closed tags)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip unclosed <think> tags (model hit token limit mid-reasoning, then
    #    continued with JSON, OR the JSON follows a truncated think block)
    if '<think>' in cleaned:
        # Take everything after the last <think> tag that was never closed
        parts = cleaned.split('<think>')
        # The JSON is most likely in the last part, after any remaining think content
        # Look for JSON start in each part from the end
        for part in reversed(parts):
            if '{' in part or '[' in part:
                cleaned = part.strip()
                break
        else:
            cleaned = parts[-1].strip()

    # 3. Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Handle unclosed fences (truncation cut off the closing ```)
        open_fence = re.search(r'```(?:json)?\s*\n?', cleaned)
        if open_fence:
            cleaned = cleaned[open_fence.end():].strip()
        # Strip any leading/trailing ``` fragments
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # 4. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 5. Find the outermost JSON object or array via bracket matching.
    #    Reasoning models (Nemotron, DeepSeek-R1, etc.) produce prose before
    #    the JSON that may contain stray '{' characters (e.g. "entities like
    #    {Alice, Bob}").  Try ALL opening-bracket positions left-to-right;
    #    stray braces in prose won't parse as valid JSON, so we naturally
    #    skip them and reach the real JSON object.
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        positions = [i for i, c in enumerate(cleaned) if c == open_char]
        for start in positions:
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(cleaned)):
                c = cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\' and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == open_char:
                    depth += 1
                elif c == close_char:
                    depth -= 1
                    if depth == 0:
                        json_str = cleaned[start:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                        break  # brackets balanced but invalid JSON → try next start

    # 6. Truncation repair: model hit token limit mid-JSON.
    #    Find the first { and attempt to close the structure.
    result = _repair_truncated_json(cleaned)
    if result is not None:
        return result

    return None


def _repair_truncated_json(text: str) -> Any:
    """Attempt to salvage a truncated JSON object/array.

    Strategy: find the outermost opening brace, take everything from there to the
    end, strip any trailing partial value (e.g. ``"type": "PEO``), then close all
    open brackets/braces. This recovers all *complete* entries in a truncated list.
    """
    start = text.find('{')
    if start == -1:
        start = text.find('[')
    if start == -1:
        return None

    fragment = text[start:]

    # Strip trailing commas
    fragment = re.sub(r',(\s*[}\]])', r'\1', fragment)

    # Try parsing as-is first (maybe it's valid after comma cleanup)
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # Truncation likely cut mid-value. Trim back to the last complete JSON element.
    # Strategy: remove the last partial key-value or array element.
    # Find the last complete "}" or "]" or quoted string followed by comma/bracket.
    # Simpler approach: walk backwards from the end, remove characters until we can
    # close the brackets and parse.

    # First, try removing everything after the last comma at depth > 0
    # (this drops the truncated element but keeps all complete ones)
    trimmed = _trim_to_last_complete_element(fragment)
    if trimmed:
        # Count unclosed brackets/braces and close them
        opens_brace = trimmed.count('{') - trimmed.count('}')
        opens_bracket = trimmed.count('[') - trimmed.count(']')
        suffix = ']' * max(opens_bracket, 0) + '}' * max(opens_brace, 0)
        candidate = trimmed + suffix
        # Fix trailing commas that may appear before the new closing brackets
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Brute force: close all open brackets/braces on the raw fragment
    opens_brace = fragment.count('{') - fragment.count('}')
    opens_bracket = fragment.count('[') - fragment.count(']')
    if opens_brace > 0 or opens_bracket > 0:
        candidate = fragment + ']' * max(opens_bracket, 0) + '}' * max(opens_brace, 0)
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _trim_to_last_complete_element(fragment: str) -> Optional[str]:
    """Walk backwards through a JSON fragment to find the end of the last
    complete element (object or value). Returns the trimmed string, or None."""
    # Find positions of all }, ], and complete quoted strings followed by
    # structural characters. We look for the last "}" that reduces depth,
    # or the last complete array element.

    # Simple heuristic: find the last occurrence of "},", "}", "],", "]",
    # or a complete value followed by "," and trim there.
    # Look for the last "}" or "]" that appears before a "," or at the end
    best = -1
    for pattern in [r'\}\s*,', r'\]\s*,', r'"\s*,', r'\d\s*,', r'true\s*,', r'false\s*,', r'null\s*,']:
        for m in re.finditer(pattern, fragment):
            end_pos = m.end() - 1  # position of the comma
            if end_pos > best:
                best = end_pos
    if best > 0:
        return fragment[:best]

    # No comma found — try last complete closing bracket
    for i in range(len(fragment) - 1, -1, -1):
        if fragment[i] in ('}', ']'):
            return fragment[:i+1]

    return None


def _unwrap_json_mode_array(result: Any) -> Any:
    """Recover top-level arrays wrapped by JSON-object constrained decoding.

    Ollama/OpenAI JSON-object mode requires a top-level object. Some older
    prompts legitimately asked for a top-level array, so models often return
    ``{"items": [...]}``, ``{"domains": [...]}``, etc. Returning the sole
    list value preserves the old callsite contract without weakening JSON mode.
    """
    if isinstance(result, dict) and len(result) == 1:
        sole_value = next(iter(result.values()))
        if isinstance(sole_value, list):
            return sole_value
    return result


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Call LLM chat completions and return the assistant message content.
    Works with both Ollama and OpenAI backends.
    """
    resolved_model = _resolve_model(model)

    # Nemotron: inject "detailed thinking off" so extraction calls emit JSON directly
    # instead of burning the budget on hidden reasoning (no-op unless NEMOTRON_THINKING!=on).
    messages = _apply_nemotron_reasoning_directive(messages, resolved_model)

    # GPT-5 / o-series reasoning models reject `max_tokens` and non-default `temperature`.
    # VLLM-hosted models always accept the standard parameters.
    is_reasoning = LLM_BACKEND == "openai" and (
        resolved_model.startswith("gpt-5")
        or resolved_model.startswith("o1")
        or resolved_model.startswith("o3")
        or resolved_model.startswith("o4")
    )

    params: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
    }
    if not is_reasoning:
        params["temperature"] = temperature
    if max_tokens is not None:
        if is_reasoning:
            # Reasoning models share max_completion_tokens between hidden reasoning
            # AND visible output. With low caps the model burns the whole budget on
            # reasoning and emits empty content. Inflate so visible output survives.
            params["max_completion_tokens"] = max(max_tokens * 4, 16384)
            # Keep reasoning lightweight so cost stays bounded for extraction tasks.
            params["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
        elif _model_is_thinking(resolved_model) and LLM_BACKEND not in ("openai",):
            # Thinking models (gemma4, deepseek-r1, qwen3, etc.) burn completion
            # tokens on hidden <think> blocks. Inflate max_tokens so the visible
            # JSON output isn't truncated after reasoning consumes the budget.
            # This applies to both Ollama and VLLM backends.
            _max_ctx = int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
            _inflated = max(max_tokens * 4, 16384)
            # Cap so prompt + max_tokens stays under the context window with a
            # 2048-token safety margin for tokenization overhead.
            params["max_tokens"] = min(_inflated, _max_ctx - 2048)
        else:
            params["max_tokens"] = max_tokens
    elif _model_is_thinking(resolved_model) and LLM_BACKEND not in ("openai",):
        # When no max_tokens is specified, thinking models still need a generous
        # budget so reasoning doesn't consume the server's default output window.
        _max_ctx = int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
        params["max_tokens"] = min(16384, _max_ctx - 2048)
    # Ollama's OpenAI-compatible endpoint supports response_format for JSON
    # mode. Pass it through for both backends, but keep arbitrary provider
    # kwargs restricted to OpenAI so local calls don't receive unknown options.
    if "response_format" in kwargs:
        params["response_format"] = kwargs["response_format"]
    if LLM_BACKEND == "openai":
        params.update(kwargs)
    # Opt-in reasoning control for hosted thinking models (e.g. Fireworks
    # nemotron: "none" cuts output tokens ~20x). Unset = provider default.
    _effort = os.getenv("LLM_REASONING_EFFORT")
    if _effort and "reasoning_effort" not in params:
        params["reasoning_effort"] = _effort

    active_client = _client_for_model(resolved_model)
    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = active_client.chat.completions.create(**params)
            _log_usage(resolved_model, response)
            choice = response.choices[0]
            content = choice.message.content
            finish = getattr(choice, "finish_reason", None)
            # Thinking models (Nemotron, deepseek-r1, qwen3…) can burn the ENTIRE
            # completion budget on hidden reasoning and get cut off before emitting
            # any visible content (finish_reason="length", content empty). Upstream
            # this surfaced as a SILENT empty extraction — the exact cause of RHF's
            # dropped gleaning triples and ~74 s "empty" calls. Detect it, grow the
            # budget, and retry within this loop instead of returning "" silently.
            if (not content) and finish == "length":
                _cur = params.get("max_tokens") or params.get("max_completion_tokens")
                # Cap must leave room for the PROMPT inside the context window:
                # a fixed margin 400'd vLLM at 129024 budget + 2049-token prompt
                # (matrix v3, doc 1). chars/3 over-estimates prompt tokens, which
                # errs on the safe side. The ceiling bounds wall time — one
                # 92-minute ladder climb showed >65K budgets cost more in
                # generation time than they recover in output; 65536 has been
                # sufficient for every rescued call so far.
                _prompt_tokens = sum(
                    len(str(m.get("content", ""))) for m in messages
                ) // 3 + 256
                _ceiling = int(os.getenv("NEMOTRON_BUDGET_CEILING", "65536"))
                _cap = min(
                    int(os.getenv("VLLM_MAX_MODEL_LEN", "32768")) - _prompt_tokens,
                    _ceiling,
                )
                if _cur and _cur < _cap and attempt < _MAX_RETRIES:
                    _bigger = min(_cur * 2, _cap)
                    if _bigger > _cur:
                        if "max_tokens" in params:
                            params["max_tokens"] = _bigger
                        if "max_completion_tokens" in params:
                            params["max_completion_tokens"] = _bigger
                        print(
                            f"  WARNING: {resolved_model} truncated before output "
                            f"(finish_reason=length) at {_cur} tokens; raising budget "
                            f"-> {_bigger} and retrying"
                        )
                        continue
                # Cannot grow further: warn loudly so this is never a silent zero.
                print(
                    f"  WARNING: {resolved_model} returned EMPTY output "
                    f"(finish_reason=length) at max budget {_cur}; reasoning overran the "
                    f"window. Shorten the prompt or raise VLLM_MAX_MODEL_LEN."
                )
            return content if content is not None else ""
        except Exception as e:
            err = str(e)
            last_error = err
            if LLM_BACKEND == "openai" and "insufficient_quota" in err:
                _QUOTA_FALLBACK = {"gpt-4o": "gpt-4o-mini", "gpt-4o-mini": "gpt-3.5-turbo"}
                if resolved_model in _QUOTA_FALLBACK:
                    fallback = _QUOTA_FALLBACK[resolved_model]
                    print(f"  Quota exceeded for {resolved_model}, retrying with {fallback}")
                    return chat_completion(messages, fallback, temperature, max_tokens, **kwargs)

            transient_markers = [
                "Connection error",
                "ConnectError",
                "ReadError",
                "RemoteProtocolError",
                "timed out",
                "Timeout",
                "connection reset",
                "connection aborted",
                "connection refused",
                "temporarily unavailable",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "server disconnected",
                "EOF",
                "502",
                "503",
                "504",
                # Rate limiting / capacity — hosted backends (Fireworks, OpenAI)
                # return these under the rapid serial calls the extraction pipeline
                # makes. Without retry, the call raises and upstream agents swallow
                # it into an empty fallback (silent zero-scored run).
                "429",
                "rate limit",
                "rate_limit",
                "too many requests",
                "overloaded",
                "capacity",
            ]
            err_lower = err.lower()
            is_transient = any(marker.lower() in err_lower for marker in transient_markers)
            should_retry = attempt < _MAX_RETRIES and is_transient
            if should_retry:
                # Exponential backoff with jitter, capped — gives Ollama time to recover
                # from tunnel drops and model reloads instead of giving up in ~12s.
                base_sleep = min(_RETRY_BACKOFF * (2 ** (attempt - 1)), _RETRY_BACKOFF_CAP)
                # If the error looks like Ollama is down, actively wait for it to come back
                # up before the next attempt so we don't burn a retry on a still-dead server.
                connection_like = any(
                    m in err_lower for m in ("connection", "refused", "reset", "eof", "server disconnected")
                )
                if connection_like and LLM_BACKEND != "openai" and not _is_fireworks_model(resolved_model):
                    _backend_label = "VLLM" if LLM_BACKEND == "vllm" else "Ollama"
                    print(
                        f"  WARNING: {_backend_label} appears unreachable on attempt {attempt}/{_MAX_RETRIES}; "
                        f"polling health endpoint for up to {_HEALTHCHECK_TIMEOUT:.0f}s before retrying"
                    )
                    deadline = time.time() + _HEALTHCHECK_TIMEOUT
                    recovered = _wait_for_ollama(deadline)
                    if recovered:
                        print(f"  {_backend_label} came back up — retrying call")
                        continue
                    print(f"  {_backend_label} still down after {_HEALTHCHECK_TIMEOUT:.0f}s; falling back to backoff sleep")
                print(
                    f"  WARNING: transient LLM failure on attempt {attempt}/{_MAX_RETRIES} "
                    f"for {resolved_model}; retrying in {base_sleep:.1f}s"
                )
                time.sleep(base_sleep)
                continue
            if any(marker in err_lower for marker in ("context length", "maximum context", "too long", "reduce the length")):
                _cur = params.get("max_tokens") or params.get("max_completion_tokens")
                if _cur and _cur > 1024:
                    _reduced = max(_cur // 2, 1024)
                    if "max_tokens" in params:
                        params["max_tokens"] = _reduced
                    if "max_completion_tokens" in params:
                        params["max_completion_tokens"] = _reduced
                    print(f"  WARNING: context-length overflow ({resolved_model}); reducing max_tokens {_cur} -> {_reduced} and retrying")
                    continue
            if "response_format" in params and any(
                marker in err_lower
                for marker in ("response_format", "unsupported", "unknown field", "invalid parameter")
            ):
                params.pop("response_format", None)
                print("  WARNING: JSON response_format unsupported by backend; retrying without constrained decoding")
                continue
            raise Exception(f"LLM API call failed ({resolved_model}): {err}")

    raise Exception(f"LLM API call failed ({resolved_model}): {last_error}")


def chat_completion_json(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    unwrap_array: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Call LLM requesting JSON output and parse the response.
    Includes robust JSON extraction that handles markdown fences,
    preamble text, thinking tags, and minor formatting issues.

    When ``unwrap_array=True``, a single-key dict-of-list response
    (e.g. ``{"items": [...]}``) is unwrapped to the inner list to
    support callsites that originally requested top-level arrays.
    Default is False so callers expecting dicts (e.g. ``{"sub_questions": [...]}``)
    are not silently stripped of their wrapper key.
    """
    # Strengthen the JSON instruction in the system prompt
    modified_messages = []
    has_system = False
    for msg in messages:
        m = dict(msg)
        if m["role"] == "system":
            has_system = True
            m["content"] = m["content"].rstrip() + "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no text before or after the JSON."
        modified_messages.append(m)

    if not has_system:
        modified_messages.insert(0, {
            "role": "system",
            "content": "You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no text before or after the JSON."
        })

    # Also reinforce in the user message
    if modified_messages and modified_messages[-1]["role"] == "user":
        content = modified_messages[-1]["content"]
        if "json" not in content.lower()[-100:]:
            modified_messages[-1]["content"] = content + "\n\nRespond with ONLY valid JSON."

    resolved_model = _resolve_model(model)

    # Detect thinking models that emit <think>...</think> before the answer.
    # On Ollama: GBNF json_object grammar blocks <think> tokens → empty content.
    # On VLLM:  json_object mode is generally safe, but thinking models may still
    #           embed reasoning in visible output; skip constrained decoding to be safe.
    _is_thinking_model = _model_is_thinking(resolved_model)

    max_retries = 4 if _is_thinking_model else 3
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        # Use json_object mode only when it won't conflict with model behavior:
        # - Always for OpenAI (native support)
        # - Never for thinking models on Ollama or VLLM: constrained decoding
        #   blocks reasoning tokens (Ollama GBNF) or strips them (VLLM), leading
        #   to empty or malformed output. Rely on _extract_json post-hoc instead.
        # - First attempt only for non-thinking Ollama/VLLM models
        if _is_fireworks_model(resolved_model):
            # Fireworks supports json_object, but thinking models (deepseek,
            # qwen3, gpt-oss…) still emit reasoning — same rule as vLLM.
            use_json_mode = (not _is_thinking_model) and attempt == 1
        elif LLM_BACKEND == "openai":
            use_json_mode = True
        elif _is_thinking_model and LLM_BACKEND in ("ollama", "vllm"):
            use_json_mode = False
        else:
            use_json_mode = (attempt == 1)

        call_kwargs = dict(kwargs)
        if use_json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        response_text = chat_completion(
            messages=modified_messages,
            model=resolved_model,
            temperature=temperature + (0.1 * (attempt - 1)),
            max_tokens=max_tokens,
            **call_kwargs,
        )
        last_response_text = response_text

        # Empty response with json_object mode → GBNF conflict, skip to unconstrained retry
        if not response_text.strip() and use_json_mode:
            print(f"  WARNING: Empty response with json_object mode (attempt {attempt}/{max_retries}); retrying without constrained decoding...")
            continue

        result = _extract_json(response_text)
        if result is not None:
            return _unwrap_json_mode_array(result) if unwrap_array else result

        if attempt < max_retries:
            print(f"  WARNING: JSON parse failed (attempt {attempt}/{max_retries}), retrying...")

    # All attempts failed - log and return empty fallback
    print(f"  WARNING: Failed to parse JSON after {max_retries} attempts.")
    print(f"  Raw (first 300 chars): {last_response_text[:300]}")
    response_text = last_response_text
    if "linked_triples" in response_text or "triples" in response_text:
        return {"linked_triples": [], "triples": []}
    elif "entities" in response_text:
        return {"entities": []}
    elif "relations" in response_text or "relation" in response_text:
        return {"relations": [], "relations_found": []}
    elif "entity_groups" in response_text:
        return {"entity_groups": []}
    else:
        return {}


def chat_completion_typed(
    messages: List[Dict[str, str]],
    schema: Type[T],
    *,
    max_validation_retries: int = 2,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> T:
    """Call the LLM, parse JSON, and validate it against a Pydantic ``schema``.

    Sits on top of :func:`chat_completion_json` (keeping all of its robust
    ``_extract_json`` / json_object / thinking-model handling) and adds schema
    validation. Lenient policy: on a :class:`ValidationError`, re-prompt the
    model with the error text and retry up to ``max_validation_retries`` times,
    then fall back to an empty, well-typed ``schema()`` instance — mirroring the
    empty-dict fallback that :func:`chat_completion_json` already returns.

    ``response_format`` is intentionally not forced here: several backends
    (Ollama GBNF, thinking models) break on constrained decoding, and Pydantic
    is the post-hoc safety net regardless of whether the backend honored a
    schema. Callers may still pass ``response_format`` through ``kwargs``.
    """
    convo: List[Dict[str, str]] = list(messages)
    last_error: Optional[ValidationError] = None

    for attempt in range(1, max_validation_retries + 2):
        raw = chat_completion_json(
            convo,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        try:
            return schema.model_validate(raw)
        except ValidationError as err:
            last_error = err
            if attempt <= max_validation_retries:
                print(
                    f"  WARNING: {schema.__name__} validation failed "
                    f"(attempt {attempt}/{max_validation_retries}); re-prompting..."
                )
                convo = convo + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous JSON failed validation: {err}. "
                            "Resend ONLY corrected JSON matching the requested schema."
                        ),
                    }
                ]

    print(
        f"  WARNING: {schema.__name__} validation failed after retries; "
        f"returning empty result. Last error: {last_error}"
    )
    return schema()


# ── Embeddings ────────────────────────────────────────────────────────
# Backend-aware default embedding model. mxbai-embed-large is asymmetric:
# retrieval queries must be prefixed, passages are embedded raw.
# VLLM: set EMBEDDING_MODEL to the embedding model served by your VLLM instance
# (e.g. "intfloat/e5-mistral-7b-instruct" or "BAAI/bge-large-en-v1.5").
# Embeddings run on OpenAI only when LLM_BACKEND=openai AND no separate
# EMBEDDING_BASE_URL is set; otherwise they hit Ollama/vLLM (mxbai default).
_EMBED_ON_OPENAI = LLM_BACKEND == "openai" and not EMBEDDING_BASE_URL
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small" if _EMBED_ON_OPENAI
    else "mxbai-embed-large",
)

_MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# nomic-embed-text is asymmetric: queries and passages take distinct task
# prefixes (Nomic spec). Skipping them costs retrieval recall.
_NOMIC_QUERY_PREFIX = "search_query: "
_NOMIC_DOC_PREFIX = "search_document: "
_QWEN_EMBED_QUERY_PREFIX = "query: "
_QWEN_EMBED_DOC_PREFIX = "passage: "


def _default_embed_max_chars(model: str) -> int:
    """Truncation cap keyed to the model's context window.

    mxbai-embed-large is 512 tokens (~1800 chars). nomic/e5/bge serve 8k windows.
    Qwen3-Embedding serves 32k tokens — cap conservatively at 24k chars.
    Override with EMBEDDING_MAX_CHARS for exact control.
    """
    m = (model or "").lower()
    if "mxbai" in m:
        return 1800
    if "qwen" in m and "embed" in m:
        return 24000
    return 8000


# Truncate conservatively to the model's window. Honor EMBEDDING_MAX_CHARS override.
_EMBED_MAX_CHARS = int(
    os.getenv("EMBEDDING_MAX_CHARS", str(_default_embed_max_chars(DEFAULT_EMBEDDING_MODEL)))
)
# GPU batching (embeddings served by vLLM) is much faster than Ollama CPU; bump
# default batch size only when embeddings actually run on vLLM. When embeddings
# are routed to a separate server (EMBEDDING_BASE_URL, e.g. Ollama), stay small.
_EMBED_ON_VLLM = LLM_BACKEND == "vllm" and not EMBEDDING_BASE_URL
_EMBED_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256" if _EMBED_ON_VLLM else "64"))


def _truncate_for_embedding(text: str) -> str:
    text = text.strip()
    if len(text) > _EMBED_MAX_CHARS:
        return text[:_EMBED_MAX_CHARS]
    return text


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Get an embedding vector for the given text."""
    return get_embeddings([text], model=model)[0]


# ── Embedding failover (owner requirement, 2026-07-13) ──────────────────
# If the primary embedding endpoint fails its whole retry ladder (e.g. the
# gpu01 Ollama wedge that stalled a 100-doc run indefinitely), fail over to a
# secondary OpenAI-compatible endpoint for the REST OF THIS PROCESS instead of
# raising. Configure with EMBEDDING_FALLBACK_BASE_URL / _API_KEY / _MODEL;
# when unset, defaults to the Fireworks embeddings API iff FIREWORKS_API_KEY
# is present. With no fallback configured, behavior is unchanged (raise).
#
# Caveat (accepted, fail-safe): vectors embedded before the switch live in a
# different embedding space than those after it. Cross-space similarities
# degrade toward "no match", so dedup/conflict-candidate consumers see fewer
# matches — never crashes or false merges.
_EMBED_FAILOVER: Dict[str, Any] = {"active": False, "client": None, "model": None}


def _fallback_embedding_config() -> Optional[Tuple[str, str, str]]:
    base = os.getenv("EMBEDDING_FALLBACK_BASE_URL")
    key = os.getenv("EMBEDDING_FALLBACK_API_KEY") or os.getenv("FIREWORKS_API_KEY")
    model = os.getenv(
        "EMBEDDING_FALLBACK_MODEL", "accounts/fireworks/models/qwen3-embedding-8b"
    )
    if not base and os.getenv("FIREWORKS_API_KEY"):
        base = "https://api.fireworks.ai/inference/v1"
    if not (base and key):
        return None
    return base, key, model


def _activate_embed_failover() -> bool:
    """Switch this process to the fallback embedding endpoint. Sticky."""
    cfg = _fallback_embedding_config()
    if cfg is None:
        return False
    base, key, model = cfg
    _EMBED_FAILOVER["client"] = OpenAI(base_url=base, api_key=key, timeout=_TIMEOUT)
    _EMBED_FAILOVER["model"] = model
    _EMBED_FAILOVER["active"] = True
    print(
        f"  WARNING: PRIMARY EMBEDDING ENDPOINT FAILED after {_MAX_RETRIES} attempts — "
        f"failing over to {base} (model={model}) for the rest of this process. "
        f"Vectors embedded before the switch are in a different space; "
        f"similarity against them degrades toward no-match (fail-safe)."
    )
    return True


def _resolved_embedding_model(model: Optional[str]) -> str:
    """The model that will actually serve the next embedding call."""
    if _EMBED_FAILOVER["active"]:
        return str(_EMBED_FAILOVER["model"])
    return model or DEFAULT_EMBEDDING_MODEL


def get_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Get embedding vectors for a batch of texts (order preserved).

    Primary endpoint with the standard retry ladder; on exhaustion, one-time
    sticky failover to the configured fallback endpoint (see above).
    """
    model = model or DEFAULT_EMBEDDING_MODEL
    cleaned = [_truncate_for_embedding(t) or " " for t in texts]
    vectors: List[List[float]] = []
    for start in range(0, len(cleaned), _EMBED_BATCH_SIZE):
        batch = cleaned[start:start + _EMBED_BATCH_SIZE]
        last_error: Optional[Exception] = None
        for source in ("primary", "fallback"):
            if source == "primary" and _EMBED_FAILOVER["active"]:
                continue  # already switched in an earlier call
            if source == "fallback" and not _EMBED_FAILOVER["active"]:
                if last_error is None or not _activate_embed_failover():
                    break  # primary succeeded, or no fallback configured
            active_client = (
                _EMBED_FAILOVER["client"] if _EMBED_FAILOVER["active"] else embed_client
            )
            active_model = (
                _EMBED_FAILOVER["model"] if _EMBED_FAILOVER["active"] else model
            )
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = active_client.embeddings.create(
                        model=active_model, input=batch
                    )
                    # API may return items out of order; sort by index.
                    items = sorted(response.data, key=lambda item: item.index)
                    vectors.extend([item.embedding for item in items])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < _MAX_RETRIES:
                        delay = min(_RETRY_BACKOFF * attempt, _RETRY_BACKOFF_CAP)
                        print(
                            f"  WARNING: Embedding call failed ({source}, attempt "
                            f"{attempt}/{_MAX_RETRIES}): {exc}; retrying in {delay:.0f}s..."
                        )
                        time.sleep(delay)
            if last_error is None:
                break
        if last_error is not None:
            raise Exception(f"Embedding API call failed: {last_error}")
    return vectors


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """Embed a retrieval query (applies the asymmetric query prefix when needed)."""
    model = _resolved_embedding_model(model)
    m = model.lower()
    if "mxbai" in m:
        text = _MXBAI_QUERY_PREFIX + text
    elif "nomic" in m:
        text = _NOMIC_QUERY_PREFIX + text
    elif "qwen" in m and "embed" in m:
        text = _QWEN_EMBED_QUERY_PREFIX + text
    return get_embedding(text, model=model)


def embed_passages(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Embed passages/documents for indexing (applies the asymmetric doc prefix when needed)."""
    model = _resolved_embedding_model(model)
    m = model.lower()
    if "nomic" in m:
        texts = [_NOMIC_DOC_PREFIX + t for t in texts]
    elif "qwen" in m and "embed" in m:
        texts = [_QWEN_EMBED_DOC_PREFIX + t for t in texts]
    return get_embeddings(texts, model=model)
