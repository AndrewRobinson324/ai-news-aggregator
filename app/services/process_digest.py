import logging
from typing import Optional

from app.agent.digest_agent import DigestAgent
from app.config import digest_llm_batch_size, llm_backend
from app.database.repository import Repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _article_ref(article: dict) -> str:
    return f"{article['type']}:{article['id']}"


def process_digests(limit: Optional[int] = None) -> dict:
    agent = DigestAgent()
    repo = Repository()

    articles = repo.get_articles_without_digest(limit=limit)
    total = len(articles)
    processed = 0
    failed = 0

    backend = llm_backend()
    if backend == "none":
        batch_size = total if total > 0 else 1
    else:
        batch_size = digest_llm_batch_size()

    logger.info(f"Starting digest processing for {total} articles (batch_size={batch_size}, backend={backend})")

    for start in range(0, total, batch_size):
        chunk = articles[start : start + batch_size]
        batch_no = start // batch_size + 1
        logger.info(f"[batch {batch_no}] Digesting {len(chunk)} articles")

        results = agent.generate_digests_batch(chunk)

        for article in chunk:
            article_type = article["type"]
            article_id = article["id"]
            ref = _article_ref(article)

            digest_result = results.get(ref)
            if not digest_result:
                logger.warning(f"✗ Missing digest for {article_type} {article_id}; skipping DB write")
                failed += 1
                continue

            try:
                repo.create_digest(
                    article_type=article_type,
                    article_id=article_id,
                    url=article["url"],
                    title=digest_result.title,
                    summary=digest_result.summary,
                    published_at=article.get("published_at"),
                )
                processed += 1
                logger.info(f"✓ Successfully created digest for {article_type} {article_id}")
            except Exception as e:
                failed += 1
                logger.error(f"✗ Error saving digest for {article_type} {article_id}: {e}")

    logger.info(f"Processing complete: {processed} processed, {failed} failed out of {total} total")

    return {
        "total": total,
        "processed": processed,
        "failed": failed,
    }


if __name__ == "__main__":
    result = process_digests()
    print(f"Total articles: {result['total']}")
    print(f"Processed: {result['processed']}")
    print(f"Failed: {result['failed']}")
