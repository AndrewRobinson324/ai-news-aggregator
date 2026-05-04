import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import (
    digest_batch_content_chars,
    gemini_model,
    llm_backend,
    ollama_digest_batch_max_tokens,
    ollama_digest_single_max_tokens,
    ollama_model,
    openai_model_digest,
)
from app.llm.ollama_client import build_ollama_openai_client
from app.llm.structured import structured_gemini, structured_ollama_chat, structured_openai

load_dotenv()

logger = logging.getLogger(__name__)


class DigestOutput(BaseModel):
    title: str
    summary: str


class BatchDigestRow(BaseModel):
    ref: str = Field(description="Exact ref from the prompt (article_type:article_id)")
    title: str
    summary: str


class BatchDigestsOutput(BaseModel):
    digests: list[BatchDigestRow] = Field(description="One digest per input article; include every ref")


PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence.

Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words) that captures the essence of the content
- Write a 2-3 sentence summary that highlights the main points and why they matter
- Focus on actionable insights and implications
- Use clear, accessible language while maintaining technical accuracy
- Avoid marketing fluff - focus on substance"""


def _article_ref(article: dict) -> str:
    return f"{article['type']}:{article['id']}"


def _template_digest(title: str, content: str, article_type: str) -> DigestOutput:
    excerpt = (content or "").strip().replace("\n", " ")
    if len(excerpt) > 800:
        excerpt = excerpt[:800] + "…"
    body = (
        f"*Template digest (no LLM).* Raw excerpt from this {article_type} item:\n\n{excerpt}"
        if excerpt
        else f"*Template digest (no LLM).* No body text was available for this {article_type} item."
    )
    clean_title = title.strip()[:200] if title else f"({article_type} item)"
    return DigestOutput(title=clean_title, summary=body)


class DigestAgent:
    def __init__(self):
        self._openai: OpenAI | None = None
        self._gemini: genai.Client | None = None
        self._ollama_client: OpenAI | None = None
        self._backend = llm_backend()
        self.openai_model = openai_model_digest()
        self.gemini_model = gemini_model()
        self.ollama_model = ollama_model()
        self.system_prompt = PROMPT
        if self._backend == "openai":
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self._backend == "gemini":
            self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self._backend == "ollama":
            self._ollama_client = build_ollama_openai_client()

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        if self._backend == "none":
            return _template_digest(title, content, article_type)

        user_prompt = f"Create a digest for this {article_type}: \n Title: {title} \n Content: {content[:8000]}"

        if self._backend == "openai":
            assert self._openai is not None
            return structured_openai(
                self._openai,
                self.openai_model,
                self.system_prompt,
                user_prompt,
                0.7,
                DigestOutput,
            )

        if self._backend == "gemini":
            assert self._gemini is not None
            return structured_gemini(
                self._gemini,
                self.gemini_model,
                self.system_prompt,
                user_prompt,
                0.7,
                DigestOutput,
            )

        assert self._ollama_client is not None
        return structured_ollama_chat(
            self._ollama_client,
            self.ollama_model,
            self.system_prompt,
            user_prompt,
            0.7,
            DigestOutput,
            max_tokens=ollama_digest_single_max_tokens(),
        )

    def generate_digests_batch(self, articles: list[dict]) -> dict[str, DigestOutput]:
        """One LLM round-trip per batch when using cloud/local LLM; template fill for missing refs."""
        if not articles:
            return {}

        if self._backend == "none":
            return {_article_ref(a): _template_digest(a["title"], a["content"], a["type"]) for a in articles}

        clip = digest_batch_content_chars()
        blocks = []
        for a in articles:
            ref = _article_ref(a)
            body = (a.get("content") or "")[:clip]
            blocks.append(f"### ref: {ref}\narticle_type: {a['type']}\ntitle: {a['title']}\ncontent:\n{body}\n")

        user_prompt = (
            "Create one digest per article below. "
            "Each digest must use the exact ref string shown after 'ref:' (format article_type:article_id).\n"
            "Include every article exactly once — do not skip any ref.\n\n" + "\n".join(blocks)
        )

        instructions = (
            self.system_prompt
            + "\n\nYou will receive multiple articles. Reply with one digest per article in the structured JSON format."
        )

        parsed: BatchDigestsOutput | None = None
        if self._backend == "openai":
            assert self._openai is not None
            parsed = structured_openai(
                self._openai,
                self.openai_model,
                instructions,
                user_prompt,
                0.7,
                BatchDigestsOutput,
            )
        elif self._backend == "gemini":
            assert self._gemini is not None
            parsed = structured_gemini(
                self._gemini,
                self.gemini_model,
                instructions,
                user_prompt,
                0.7,
                BatchDigestsOutput,
            )
        else:
            assert self._ollama_client is not None
            parsed = structured_ollama_chat(
                self._ollama_client,
                self.ollama_model,
                instructions,
                user_prompt,
                0.7,
                BatchDigestsOutput,
                max_tokens=ollama_digest_batch_max_tokens(),
            )

        out: dict[str, DigestOutput] = {}
        llm_keys: set[str] = set()
        if parsed and parsed.digests:
            for row in parsed.digests:
                key = row.ref.strip()
                llm_keys.add(key)
                out[key] = DigestOutput(title=row.title, summary=row.summary)

        for a in articles:
            ref = _article_ref(a)
            if ref not in out:
                out[ref] = _template_digest(a["title"], a["content"], a["type"])

        if self._backend == "ollama":
            for a in articles:
                ref = _article_ref(a)
                if ref not in llm_keys:
                    one = self.generate_digest(a["title"], a["content"], a["type"])
                    if one:
                        out[ref] = one
                        logger.info("Ollama: filled digest for %s after batch omitted it", ref)

        return out
