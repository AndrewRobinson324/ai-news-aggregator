import logging
import os
import time
from typing import List

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import (
    curator_llm_max_digests,
    curator_prompt_summary_chars,
    gemini_model,
    llm_backend,
    ollama_curator_max_tokens,
    ollama_model,
    openai_model_curator,
)
from app.llm.ollama_client import build_ollama_openai_client
from app.llm.structured import structured_gemini, structured_ollama_chat, structured_openai

load_dotenv()

logger = logging.getLogger(__name__)


class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0)
    rank: int = Field(description="Rank position (1 = most relevant)", ge=1)
    reasoning: str = Field(description="Brief explanation of why this article is ranked here")


class RankedDigestList(BaseModel):
    articles: List[RankedArticle] = Field(description="List of ranked articles")


CURATOR_PROMPT = """You are an expert AI news curator specializing in personalized content ranking for AI professionals.

Your role is to analyze and rank AI-related news articles, research papers, and video content based on a user's specific profile, interests, and background.

Ranking Criteria:
1. Relevance to user's stated interests and background
2. Technical depth and practical value
3. Novelty and significance of the content
4. Alignment with user's expertise level
5. Actionability and real-world applicability

Scoring Guidelines:
- 9.0-10.0: Highly relevant, directly aligns with user interests, significant value
- 7.0-8.9: Very relevant, strong alignment with interests, good value
- 5.0-6.9: Moderately relevant, some alignment, decent value
- 3.0-4.9: Somewhat relevant, limited alignment, lower value
- 0.0-2.9: Low relevance, minimal alignment, little value

Rank articles from most relevant (rank 1) to least relevant. Ensure each article has a unique rank."""


def _sort_key(d: dict) -> float:
    ca = d.get("created_at")
    if ca is None:
        return 0.0
    ts = getattr(ca, "timestamp", None)
    return float(ts()) if callable(ts) else 0.0


def _clip_for_prompt(text: str, max_chars: int) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _rank_by_recency(digests: List[dict]) -> List[RankedArticle]:
    ordered = sorted(digests, key=_sort_key, reverse=True)
    out: List[RankedArticle] = []
    n = len(ordered)
    for i, d in enumerate(ordered):
        score = max(5.0, 10.0 - i * (5.0 / max(n, 1)))
        out.append(
            RankedArticle(
                digest_id=d["id"],
                relevance_score=round(score, 1),
                rank=i + 1,
                reasoning="Ordered by digest time (newest first). Template mode — no LLM ranking.",
            )
        )
    return out


class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self._openai: OpenAI | None = None
        self._gemini: genai.Client | None = None
        self._ollama_client: OpenAI | None = None
        self._backend = llm_backend()
        self.openai_model = openai_model_curator()
        self.gemini_model = gemini_model()
        self.ollama_model = ollama_model()
        self.system_prompt = self._build_system_prompt()
        if self._backend == "openai":
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self._backend == "gemini":
            self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self._backend == "ollama":
            self._ollama_client = build_ollama_openai_client()

    def _build_system_prompt(self) -> str:
        interests = "\n".join(f"- {interest}" for interest in self.user_profile["interests"])
        preferences = self.user_profile["preferences"]
        pref_text = "\n".join(f"- {k}: {v}" for k, v in preferences.items())

        return f"""{CURATOR_PROMPT}

User Profile:
Name: {self.user_profile["name"]}
Background: {self.user_profile["background"]}
Expertise Level: {self.user_profile["expertise_level"]}

Interests:
{interests}

Preferences:
{pref_text}"""

    def rank_digests(self, digests: List[dict]) -> List[RankedArticle]:
        if not digests:
            return []

        if self._backend == "none":
            return _rank_by_recency(digests)

        summary_chars = curator_prompt_summary_chars(self._backend)
        ordered_full = sorted(digests, key=_sort_key, reverse=True)
        max_llm = curator_llm_max_digests(self._backend)
        tail: List[dict] = []
        working = ordered_full
        if max_llm is not None and len(ordered_full) > max_llm:
            working = ordered_full[:max_llm]
            tail = ordered_full[max_llm:]
            logger.info(
                "Curator: LLM ranks newest %d digest(s); %d older digest(s) appended by recency after",
                len(working),
                len(tail),
            )

        digest_list = "\n\n".join(
            [
                f"ID: {d['id']}\nTitle: {d['title']}\nSummary: {_clip_for_prompt(str(d.get('summary')), summary_chars)}\nType: {d['article_type']}"
                for d in working
            ]
        )

        n = len(working)
        user_prompt = f"""Rank these {n} AI news digests based on the user profile:

{digest_list}

Provide a relevance score (0.0-10.0) and rank (1-{n}) for each article, ordered from most to least relevant."""

        logger.info("Curator: calling LLM backend=%s on %d digest(s)…", self._backend, n)
        t0 = time.perf_counter()

        ranked_list: RankedDigestList | None = None
        if self._backend == "openai":
            assert self._openai is not None
            ranked_list = structured_openai(
                self._openai,
                self.openai_model,
                self.system_prompt,
                user_prompt,
                0.3,
                RankedDigestList,
            )
        elif self._backend == "gemini":
            assert self._gemini is not None
            ranked_list = structured_gemini(
                self._gemini,
                self.gemini_model,
                self.system_prompt,
                user_prompt,
                0.3,
                RankedDigestList,
            )
        else:
            assert self._ollama_client is not None
            ranked_list = structured_ollama_chat(
                self._ollama_client,
                self.ollama_model,
                self.system_prompt,
                user_prompt,
                0.3,
                RankedDigestList,
                max_tokens=ollama_curator_max_tokens(),
            )

        logger.info("Curator: LLM returned in %.1fs", time.perf_counter() - t0)

        articles = ranked_list.articles if ranked_list else []
        if not articles:
            logger.warning("LLM ranking returned no results; falling back to recency order")
            return _rank_by_recency(digests)

        articles.sort(key=lambda a: a.rank)
        articles = [a.model_copy(update={"rank": i}) for i, a in enumerate(articles, start=1)]

        if tail:
            base = len(articles)
            for i, d in enumerate(tail):
                articles.append(
                    RankedArticle(
                        digest_id=d["id"],
                        relevance_score=max(2.0, 4.5 - i * 0.2),
                        rank=base + i + 1,
                        reasoning="Older than curated batch — ordered by recency.",
                    )
                )

        return articles
