#!/usr/bin/env python3
"""Run a command with the project routed to Fireworks + Jev credentials.

The local ``.env`` mixes a gpu02 block and a Fireworks block (and stores the
Jev key as ``Jev_API_KEY`` with a leading space), so ``source .env`` is unsafe.
This launcher reads it with python-dotenv, pins a Fireworks-only config, and
``exec``s the given command. Secrets never touch stdout.

Usage: python scripts/jev/fw_exec.py <cmd> [args...]
"""

import os
import sys

from dotenv import dotenv_values

FW_BASE = "https://api.fireworks.ai/inference/v1"
FW_CHAT = os.getenv("JEV_EXP_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-120b")
FW_EMBED = "accounts/fireworks/models/qwen3-embedding-8b"


def main() -> None:
    values = {k: (v or "").strip() for k, v in dotenv_values(".env").items()}
    fw_key = values.get("FIREWORKS_API_KEY") or values.get("VLLM_API_KEY", "")
    jev_key = values.get("TYPESAFE_API_KEY") or values.get("Jev_API_KEY", "")
    env = dict(os.environ)
    env.update(
        {
            "LLM_BACKEND": "vllm",
            "VLLM_BASE_URL": FW_BASE,
            "VLLM_API_KEY": fw_key,
            "FIREWORKS_API_KEY": fw_key,
            "LLM_DEFAULT_MODEL": FW_CHAT,
            "EMBEDDING_BASE_URL": FW_BASE,
            "EMBEDDING_API_KEY": values.get("EMBEDDING_API_KEY") or fw_key,
            "EMBEDDING_MODEL": FW_EMBED,
            "TYPESAFE_API_KEY": jev_key,
            "VLLM_MAX_MODEL_LEN": "131072",
            "PYTHONPATH": ".",
        }
    )
    env.setdefault("LLM_USAGE_LOG", "evaluation/results/jev/fw_usage.jsonl")
    os.makedirs("evaluation/results/jev", exist_ok=True)
    os.execvpe(sys.argv[1], sys.argv[1:], env)


if __name__ == "__main__":
    main()
