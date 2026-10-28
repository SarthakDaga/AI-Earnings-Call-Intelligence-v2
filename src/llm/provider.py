"""
src/llm/provider.py

LLM provider interface (S4). Selected by env var, all options free:

  ollama         local models via Ollama (default). No key, no cost.
  openai_compat  any OpenAI-compatible endpoint. Several vendors offer free tiers
                 (Groq, Google AI Studio, OpenRouter ":free" models). Free-tier
                 limits change, so check the vendor before relying on one.
  fake           scripted responses for tests.

Plain HTTP via `requests` keeps this independent of SDK version changes.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol

import requests

from src import config


class LLMError(RuntimeError):
    """kind: connection | model_missing | http | parse"""

    def __init__(self, message: str, kind: str = "http"):
        super().__init__(message)
        self.kind = kind


class LLM(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512, json_mode: bool = False) -> str: ...


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int | None = None):
        self.model = model or config.LLM_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout or config.LLM_TIMEOUT

    def chat(self, messages, *, temperature=0.0, max_tokens=512, json_mode=False) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}}
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(f"Could not connect to Ollama at {self.host}. Run: ollama serve", "connection") from e
        except requests.Timeout as e:
            raise LLMError("Ollama request timed out", "http") from e
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not installed. Run: ollama pull {self.model}", "model_missing")
        if not r.ok:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:200]}", "http")
        return r.json().get("message", {}).get("content", "")

    def status(self) -> dict:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            return {"available": False, "reason": f"Ollama not reachable: {e}", "installed_models": []}
        base = self.model.split(":")[0]
        found = any(self.model == n or (":" not in self.model and n.startswith(base)) for n in names)
        return {"available": found, "installed_models": names,
                "reason": None if found else f"Run: ollama pull {self.model}"}


class OpenAICompatLLM:
    provider = "openai_compat"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: int | None = None):
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else config.OPENAI_API_KEY
        self.timeout = timeout or config.LLM_TIMEOUT

    def chat(self, messages, *, temperature=0.0, max_tokens=512, json_mode=False) -> str:
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(f"Could not connect to {self.base_url}", "connection") from e
        except requests.Timeout as e:
            raise LLMError("LLM request timed out", "http") from e
        if r.status_code == 429:
            raise LLMError("Rate limited by the provider (free tiers have low limits). Retry shortly.", "http")
        if not r.ok:
            raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:200]}", "http")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("Unexpected response shape from provider", "parse") from e

    def status(self) -> dict:
        ok = bool(self.api_key)
        return {"available": ok, "installed_models": [self.model],
                "reason": None if ok else "OPENAI_API_KEY is not set"}


class FakeLLM:
    """Deterministic stub. `script` is a list of responses or a callable(messages)->str."""
    provider = "fake"

    def __init__(self, script: list[str] | Callable[[list[dict]], str] | None = None, model: str = "fake"):
        self.model = model
        self.script = script if script is not None else ["INSUFFICIENT_EVIDENCE"]
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=512, json_mode=False) -> str:
        self.calls.append(messages)
        if callable(self.script):
            return self.script(messages)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        return self.script[i]

    def status(self) -> dict:
        return {"available": True, "installed_models": ["fake"], "reason": None}


def get_llm(provider: str | None = None, model: str | None = None, role: str = "gen"):
    """role='judge' picks the JUDGE_* settings (a different model family than the generator)."""
    if role == "judge":
        provider = provider or config.JUDGE_PROVIDER
        model = model or config.JUDGE_MODEL
    else:
        provider = provider or config.LLM_PROVIDER
        model = model or config.LLM_MODEL
    if provider == "ollama":
        return OllamaLLM(model)
    if provider == "openai_compat":
        if role == "judge":
            return OpenAICompatLLM(model, config.JUDGE_BASE_URL, config.JUDGE_API_KEY)
        return OpenAICompatLLM(model)
    if provider == "fake":
        return FakeLLM(model=model)
    raise ValueError(f"Unknown LLM provider '{provider}' (ollama | openai_compat | fake)")


def extract_json(text: str):
    """Tolerant JSON extraction: handles code fences, preambles, trailing chatter."""
    if not text:
        raise LLMError("empty response", "parse")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except ValueError:
                        break
        start = cleaned.find("{", start + 1)
    raise LLMError(f"no JSON object found in: {text[:120]!r}", "parse")
