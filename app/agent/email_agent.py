import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import use_openai_llm

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
        self._client: OpenAI | None = None
        self.model = "gpt-4o-mini"
        if use_openai_llm():
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_introduction(self, ranked_articles: List) -> EmailIntroduction:
        current_date = datetime.now().strftime("%B %d, %Y")
        name = self.user_profile["name"]

        if not ranked_articles:
            return EmailIntroduction(
                greeting=f"Hey {name}, here is your daily digest of AI news for {current_date}.",
                introduction="No articles were ranked today.",
            )

        if not use_openai_llm():
            n = len(ranked_articles[:10])
            return EmailIntroduction(
                greeting=f"Hey {name}, here is your AI news roundup for {current_date}.",
                introduction=(
                    f"Below are up to {n} items from your configured sources. "
                    "Summaries use template excerpts (set USE_OPENAI=true with a valid API key for LLM-written digests)."
                ),
            )

        assert self._client is not None
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

        try:
            response = self._client.responses.parse(
                model=self.model,
                instructions=EMAIL_PROMPT,
                temperature=0.7,
                input=user_prompt,
                text_format=EmailIntroduction,
            )

            intro = response.output_parsed
            if intro and not intro.greeting.startswith(f"Hey {name}"):
                intro.greeting = f"Hey {name}, here is your daily digest of AI news for {current_date}."

            return intro
        except Exception as e:
            print(f"Error generating introduction: {e}")
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
