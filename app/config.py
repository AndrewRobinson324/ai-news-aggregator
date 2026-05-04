"""App configuration: sources and optional LLM toggle."""

import os

# YouTube channel IDs for RSS: https://www.youtube.com/feeds/videos.xml?channel_id=<ID>

YOUTUBE_CHANNELS = [
    "UCawZsQWqfGSbCI5yjkdVkTA",  # Matthew Berman
    "UCn8ujwUInbJkBhffxqAPBVQ",  # Dave Ebbelaar
]


def use_openai_llm() -> bool:
    """Use OpenAI only when explicitly enabled and a key is set."""
    enabled = os.getenv("USE_OPENAI", "false").strip().lower() in ("1", "true", "yes")
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return enabled and bool(key)
