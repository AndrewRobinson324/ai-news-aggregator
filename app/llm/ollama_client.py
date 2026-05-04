"""OpenAI SDK client pointed at Ollama's native `/v1` compatibility layer."""

import os

from openai import OpenAI

from app.config import ollama_base_url, ollama_http_timeout_seconds


def build_ollama_openai_client() -> OpenAI:
    return OpenAI(
        base_url=f"{ollama_base_url()}/v1",
        api_key=os.getenv("OLLAMA_API_KEY") or "ollama",
        timeout=ollama_http_timeout_seconds(),
    )
