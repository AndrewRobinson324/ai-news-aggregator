"""App configuration: sources and optional LLM backends."""

import os
from typing import Literal

# YouTube channel IDs for RSS: https://www.youtube.com/feeds/videos.xml?channel_id=<ID>

YOUTUBE_CHANNELS = [
    "UCawZsQWqfGSbCI5yjkdVkTA",  # Matthew Berman
    "UCn8ujwUInbJkBhffxqAPBVQ",  # Dave Ebbelaar
]

LLMBackend = Literal["none", "openai", "gemini", "ollama"]


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _openai_ready() -> bool:
    return _truthy("USE_OPENAI") and bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _gemini_ready() -> bool:
    return _truthy("USE_GEMINI") and bool((os.getenv("GEMINI_API_KEY") or "").strip())


def _ollama_ready() -> bool:
    """Ollama is enabled when USE_OLLAMA is true; model defaults to ollama_model() if unset."""
    return _truthy("USE_OLLAMA")


def llm_backend() -> LLMBackend:
    """Active structured-LLM backend. Set LLM_PROVIDER when more than one backend is configured."""
    pref = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    o = _openai_ready()
    g = _gemini_ready()
    ol = _ollama_ready()

    if pref == "openai":
        return "openai" if o else "none"
    if pref == "gemini":
        return "gemini" if g else "none"
    if pref == "ollama":
        return "ollama" if ol else "none"
    if pref in ("none", "off", "false"):
        return "none"

    configured = sum(bool(x) for x in (o, g, ol))
    if configured > 1:
        return "none"

    if ol:
        return "ollama"
    if g:
        return "gemini"
    if o:
        return "openai"
    return "none"


def llm_provider_conflict_message() -> str | None:
    """Warn when multiple LLM backends are enabled but LLM_PROVIDER does not pick one."""
    if (os.getenv("LLM_PROVIDER") or "").strip():
        return None
    configured = sum(bool(x) for x in (_openai_ready(), _gemini_ready(), _ollama_ready()))
    if configured > 1:
        return (
            "Multiple LLM backends are configured (OpenAI / Gemini / Ollama). "
            "Set LLM_PROVIDER=gemini, LLM_PROVIDER=openai, or LLM_PROVIDER=ollama."
        )
    return None


def use_openai_llm() -> bool:
    """True when OpenAI is the active backend (backward-compatible check)."""
    return llm_backend() == "openai"


def gemini_model() -> str:
    return (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()


def openai_model_digest() -> str:
    return (os.getenv("OPENAI_MODEL_DIGEST") or "gpt-4o-mini").strip()


def openai_model_curator() -> str:
    return (os.getenv("OPENAI_MODEL_CURATOR") or "gpt-4.1").strip()


def openai_model_email() -> str:
    return (os.getenv("OPENAI_MODEL_EMAIL") or "gpt-4o-mini").strip()


def ollama_base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str:
    return (os.getenv("OLLAMA_MODEL") or "llama3.2").strip()


def ollama_http_timeout_seconds() -> float:
    """Long reads — batched digests on CPU can take several minutes."""
    raw = (os.getenv("OLLAMA_TIMEOUT") or "600").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def digest_llm_batch_size() -> int:
    raw = (os.getenv("DIGEST_LLM_BATCH_SIZE") or "8").strip()
    try:
        n = int(raw)
    except ValueError:
        return 8
    return max(1, min(n, 24))


def digest_batch_content_chars() -> int:
    raw = (os.getenv("DIGEST_BATCH_CONTENT_CHARS") or "3500").strip()
    try:
        n = int(raw)
    except ValueError:
        return 3500
    return max(500, min(n, 12000))


def curator_llm_max_digests(backend: str) -> int | None:
    """Cap items sent to curator LLM (critical for slow local models). None = no cap."""
    raw = (os.getenv("CURATOR_LLM_MAX_DIGESTS") or "").strip().lower()
    if raw in ("none", "unlimited"):
        return None
    if raw:
        try:
            return max(3, int(raw))
        except ValueError:
            pass
    return 12 if backend == "ollama" else None


def curator_prompt_summary_chars(backend: str) -> int:
    raw = (os.getenv("CURATOR_PROMPT_SUMMARY_CHARS") or "").strip()
    if raw:
        try:
            return max(80, int(raw))
        except ValueError:
            pass
    return 480 if backend == "ollama" else 1500


def ollama_curator_max_tokens() -> int:
    raw = (os.getenv("OLLAMA_CURATOR_MAX_TOKENS") or "6144").strip()
    try:
        return max(1024, int(raw))
    except ValueError:
        return 6144


def ollama_digest_batch_max_tokens() -> int:
    """Completion budget for batched digest JSON (many articles)."""
    raw = (os.getenv("OLLAMA_DIGEST_BATCH_MAX_TOKENS") or "12288").strip()
    try:
        return max(4096, int(raw))
    except ValueError:
        return 12288


def ollama_digest_single_max_tokens() -> int:
    """Per-article digest fallback after an incomplete batch."""
    raw = (os.getenv("OLLAMA_DIGEST_SINGLE_MAX_TOKENS") or "2048").strip()
    try:
        return max(512, int(raw))
    except ValueError:
        return 2048
