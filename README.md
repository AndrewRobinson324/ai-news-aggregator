# Andrew's AI News Aggregator

This is my personal AI news pipeline.

The project collects recent AI content from a small set of trusted sources, stores it in PostgreSQL, builds digest entries, ranks them for a daily email, and sends the result to my inbox.

## What This Project Does

On each run, the pipeline does five steps:

1. Scrape latest content:
   - YouTube channel RSS feeds
   - OpenAI news RSS
   - Anthropic news/research/engineering RSS
2. Process source content:
   - Fetch Anthropic article pages and convert to markdown
   - Fetch YouTube transcripts for new videos
3. Create digest records in the database
4. Rank and format a "top N" email digest
5. Send email via Gmail SMTP

By default, this repo currently runs in a no-paid-LLM mode:
- `USE_OPENAI=false` uses template summaries/ranking (no OpenAI API calls required).
- If I later enable `USE_OPENAI=true` with a valid key and quota, the OpenAI-based agents are used automatically.

## Stack

- Python 3.12
- SQLAlchemy + PostgreSQL
- Docker Compose (local Postgres)
- RSS + transcript + markdown extraction
- Gmail SMTP for delivery

## Quick Start

1. Create and activate a virtualenv.
2. Install dependencies.
3. Start PostgreSQL in Docker.
4. Configure `.env`.
5. Create database tables.
6. Run the pipeline.

```bash
cd ~/projects/ai-news-aggregator
uv pip install -e .
docker compose -f docker/docker-compose.yml up -d
cp app/example.env .env
.venv/bin/python -m app.database.create_tables
.venv/bin/python main.py 168 10
```

## Environment Variables

The app reads `.env` from repo root.

- `USE_OPENAI` — `false` by default
- `OPENAI_API_KEY` — only needed when `USE_OPENAI=true`
- `MY_EMAIL` — recipient/sender Gmail address
- `APP_PASSWORD` — Gmail App Password (not normal account password)
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`

## Project Layout

- `main.py` - CLI entrypoint for daily pipeline
- `app/daily_runner.py` - orchestrates end-to-end workflow
- `app/runner.py` - source scraping and DB writes
- `app/scrapers/` - YouTube, OpenAI, Anthropic source adapters
- `app/database/` - SQLAlchemy models, session, repository, table creation
- `app/services/` - processing steps and SMTP email sender
- `app/agent/` - digest/ranking/email logic (OpenAI or template mode)
- `app/profiles/` - personalization profile used in curation/email
- `docker/docker-compose.yml` - local PostgreSQL service
