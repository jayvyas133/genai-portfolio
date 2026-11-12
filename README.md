# AI Portfolio

Five production-style AI projects built on a shared platform layer.

| # | Project | Stack | Summary |
|---|---------|-------|---------|
| 0 | [aicore](packages/aicore/) | Python | Shared platform layer: LLM clients, config, tracing, evals |
| 1 | [Document Intelligence](projects/01-doc-intelligence/) | Flask | _TBD_ |
| 2 | [Text-to-SQL Agent](projects/02-text-to-sql-agent/) | Flask + LangGraph | _TBD_ |
| 3 | [SQL SLM Fine-tune](projects/03-sql-slm-finetune/) | Training + vLLM | _TBD_ |
| 4 | [Deep Research](projects/04-deep-research/) | FastAPI | _TBD_ |
| 5 | [Indic Voice Agent](projects/05-indic-voice-agent/) | FastAPI | _TBD_ |

## Quickstart

```bash
uv sync                                   # install all workspace members
docker compose -f infra/docker-compose.yml up -d   # Postgres+pgvector, Redis, Langfuse
uv run pytest                             # run tests
```
