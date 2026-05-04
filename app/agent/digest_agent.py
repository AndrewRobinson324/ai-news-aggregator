import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from app.config import use_openai_llm

load_dotenv()


class DigestOutput(BaseModel):
    title: str
    summary: str


PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence.

Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words) that captures the essence of the content
- Write a 2-3 sentence summary that highlights the main points and why they matter
- Focus on actionable insights and implications
- Use clear, accessible language while maintaining technical accuracy
- Avoid marketing fluff - focus on substance"""


def _template_digest(title: str, content: str, article_type: str) -> DigestOutput:
    excerpt = (content or "").strip().replace("\n", " ")
    if len(excerpt) > 800:
        excerpt = excerpt[:800] + "…"
    body = (
        f"*Template digest (no OpenAI).* Raw excerpt from this {article_type} item:\n\n{excerpt}"
        if excerpt
        else f"*Template digest (no OpenAI).* No body text was available for this {article_type} item."
    )
    clean_title = title.strip()[:200] if title else f"({article_type} item)"
    return DigestOutput(title=clean_title, summary=body)


class DigestAgent:
    def __init__(self):
        self._client: OpenAI | None = None
        self.model = "gpt-4o-mini"
        self.system_prompt = PROMPT
        if use_openai_llm():
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        if not use_openai_llm():
            return _template_digest(title, content, article_type)

        assert self._client is not None
        try:
            user_prompt = f"Create a digest for this {article_type}: \n Title: {title} \n Content: {content[:8000]}"

            response = self._client.responses.parse(
                model=self.model,
                instructions=self.system_prompt,
                temperature=0.7,
                input=user_prompt,
                text_format=DigestOutput,
            )

            return response.output_parsed
        except Exception as e:
            print(f"Error generating digest: {e}")
            return None
