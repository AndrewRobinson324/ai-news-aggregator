import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import gemini_model, llm_backend, ollama_model, openai_model_email
from app.llm.ollama_client import build_ollama_openai_client
from app.llm.structured import structured_gemini, structured_ollama_chat, structured_openai

load_dotenv()


class EmailIntroduction(BaseModel):
    greeting: str = Field(description="Personalized greeting with user's name and date")
    introduction: str = Field(description="2-3 sentence overview of what's in the top 10 ranked articles")


class RankedArticleDetail(BaseModel):
    digest_id: str
    rank: int
    relevance_score: float
    title: str
    summary: str
    url: str
    article_type: str
    reasoning: Optional[str] = None


class EmailDigestResponse(BaseModel):
    introduction: EmailIntroduction
    articles: List[RankedArticleDetail]
    total_ranked: int
    top_n: int

    def to_markdown(self) -> str:
        markdown = f"{self.introduction.greeting}\n\n"
        markdown += f"{self.introduction.introduction}\n\n"
        markdown += "---\n\n"

        for article in self.articles:
            markdown += f"## {article.title}\n\n"
            markdown += f"{article.summary}\n\n"
            markdown += f"[Read more →]({article.url})\n\n"
            markdown += "---\n\n"

        return markdown


class EmailDigest(BaseModel):
    introduction: EmailIntroduction
    ranked_articles: List[dict] = Field(description="Top 10 ranked articles with their details")


EMAIL_PROMPT = """You are an expert email writer specializing in creating engaging, personalized AI news digests.

Your role is to write a warm, professional introduction for a daily AI news digest email that:
- Greets the user by name
- Includes the current date
- Provides a brief, engaging overview of what's coming in the top 10 ranked articles
- Highlights the most interesting or important themes
- Sets expectations for the content ahead

Keep it concise (2-3 sentences for the introduction), friendly, and professional."""


class EmailAgent:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self._openai: OpenAI | None = None
        self._gemini: genai.Client | None = None
        self._ollama_client: OpenAI | None = None
        self._backend = llm_backend()
        self.openai_model = openai_model_email()
        self.gemini_model = gemini_model()
        self.ollama_model = ollama_model()
        if self._backend == "openai":
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self._backend == "gemini":
            self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self._backend == "ollama":
            self._ollama_client = build_ollama_openai_client()

    def generate_introduction(self, ranked_articles: List) -> EmailIntroduction:
        current_date = datetime.now().strftime("%B %d, %Y")
        name = self.user_profile["name"]

        if not ranked_articles:
            return EmailIntroduction(
                greeting=f"Hey {name}, here is your daily digest of AI news for {current_date}.",
                introduction="No articles were ranked today.",
            )

        if self._backend == "none":
            n = len(ranked_articles[:10])
            return EmailIntroduction(
                greeting=f"Hey {name}, here is your AI news roundup for {current_date}.",
                introduction=(
                    f"Below are up to {n} items from your configured sources. "
                    "Summaries use template excerpts, or enable an LLM: USE_GEMINI, USE_OPENAI, or USE_OLLAMA."
                ),
            )

        top_articles = ranked_articles[:10]
        article_summaries = "\n".join(
            [
                f"{idx + 1}. {article.title if hasattr(article, 'title') else article.get('title', 'N/A')} (Score: {article.relevance_score if hasattr(article, 'relevance_score') else article.get('relevance_score', 0):.1f}/10)"
                for idx, article in enumerate(top_articles)
            ]
        )

        user_prompt = f"""Create an email introduction for {name} for {current_date}.

Top 10 ranked articles:
{article_summaries}

Generate a greeting and introduction that previews these articles."""

        intro: EmailIntroduction | None = None
        if self._backend == "openai":
            assert self._openai is not None
            intro = structured_openai(
                self._openai,
                self.openai_model,
                EMAIL_PROMPT,
                user_prompt,
                0.7,
                EmailIntroduction,
            )
        elif self._backend == "gemini":
            assert self._gemini is not None
            intro = structured_gemini(
                self._gemini,
                self.gemini_model,
                EMAIL_PROMPT,
                user_prompt,
                0.7,
                EmailIntroduction,
            )
        else:
            assert self._ollama_client is not None
            intro = structured_ollama_chat(
                self._ollama_client,
                self.ollama_model,
                EMAIL_PROMPT,
                user_prompt,
                0.7,
                EmailIntroduction,
            )

        if intro and not intro.greeting.startswith(f"Hey {name}"):
            intro.greeting = f"Hey {name}, here is your daily digest of AI news for {current_date}."

        if intro:
            return intro

        return EmailIntroduction(
            greeting=f"Hey {name}, here is your daily digest of AI news for {current_date}.",
            introduction="Here are the top AI news articles ranked by relevance to your interests.",
        )

    def create_email_digest(self, ranked_articles: List[dict], limit: int = 10) -> EmailDigest:
        top_articles = ranked_articles[:limit]
        introduction = self.generate_introduction(top_articles)

        return EmailDigest(
            introduction=introduction,
            ranked_articles=top_articles,
        )

    def create_email_digest_response(
        self,
        ranked_articles: List[RankedArticleDetail],
        total_ranked: int,
        limit: int = 10,
    ) -> EmailDigestResponse:
        top_articles = ranked_articles[:limit]
        introduction = self.generate_introduction(top_articles)

        return EmailDigestResponse(
            introduction=introduction,
            articles=top_articles,
            total_ranked=total_ranked,
            top_n=limit,
        )
