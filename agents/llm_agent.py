"""
LLM Agent — provider abstraction for Groq, Gemini, and OpenRouter.
Switch providers via environment variables only.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import streamlit as st

PROVIDER_DEFAULTS = {
    "groq": {
        "model_env": "GROQ_MODEL",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": None,
    },
    "gemini": {
        "model_env": "GEMINI_MODEL",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-1.5-flash",
        "base_url": None,
    },
    "openrouter": {
        "model_env": "OPENROUTER_MODEL",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.2-3b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
    },
}


class LLMError(Exception):
    """Raised when LLM calls fail — never use fake fallbacks."""

    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


def _get_secret(key: str) -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, "")


def get_llm_config() -> dict[str, str]:
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        provider = "groq"
    cfg = PROVIDER_DEFAULTS[provider]
    model = os.environ.get(cfg["model_env"], cfg["default_model"])
    api_key = _get_secret(cfg["key_env"])
    return {"provider": provider, "model": model, "api_key": api_key}


def is_configured() -> bool:
    cfg = get_llm_config()
    return bool(cfg["api_key"] and len(cfg["api_key"]) > 8)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Invalid JSON from LLM: {e}")


def _call_groq(prompt: str, model: str, api_key: str) -> str:
    try:
        from groq import Groq
    except ImportError as e:
        raise LLMError("Groq SDK not installed. Run: pip install groq") from e

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or "rate" in msg or "429" in msg:
            raise LLMError(f"Groq API quota or rate limit exceeded: {e}", provider="groq")
        raise LLMError(f"Groq API error: {e}", provider="groq")


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text or ""
    except ImportError:
        pass

    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        if hasattr(response, "text"):
            return response.text or ""
        return str(response)
    except ImportError as e:
        raise LLMError(
            "Gemini SDK not installed. Run: pip install google-generativeai or google-genai"
        ) from e
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or "429" in msg:
            raise LLMError(f"Gemini API quota exceeded: {e}", provider="gemini")
        raise LLMError(f"Gemini API error: {e}", provider="gemini")


def _call_openrouter(prompt: str, model: str, api_key: str) -> str:
    import urllib.error
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://insightpilot.local",
            "X-Title": "InsightPilot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if e.code == 429:
            raise LLMError(f"OpenRouter rate limit exceeded: {body}", provider="openrouter")
        raise LLMError(f"OpenRouter API error ({e.code}): {body}", provider="openrouter")
    except Exception as e:
        raise LLMError(f"OpenRouter API error: {e}", provider="openrouter")


def call_llm(prompt: str, json_mode: bool = False, retries: int = 2) -> str | dict[str, Any]:
    """
    Unified LLM entry point. Raises LLMError on failure — no silent fallbacks.
    """
    cfg = get_llm_config()
    if not is_configured():
        raise LLMError(
            f"{cfg['provider'].title()} API key not configured. "
            f"Set {PROVIDER_DEFAULTS[cfg['provider']]['key_env']} in .env or Streamlit secrets."
        )

    provider = cfg["provider"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            if provider == "groq":
                text = _call_groq(prompt, model, api_key)
            elif provider == "gemini":
                text = _call_gemini(prompt, model, api_key)
            elif provider == "openrouter":
                text = _call_openrouter(prompt, model, api_key)
            else:
                raise LLMError(f"Unknown provider: {provider}")

            if json_mode:
                return _parse_json_response(text)
            return text.strip()

        except LLMError as e:
            last_error = e
            if attempt < retries and e.retryable:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise LLMError(f"{provider.title()} API error: {e}", provider=provider) from e

    raise LLMError(str(last_error) if last_error else "Unknown LLM error", provider=provider)
