# AI Portfolio

Three production-style GenAI (Generative AI) projects built on a shared
platform layer. Together they go from retrieval, to coordinating agents,
to a conversational agent.

| # | Project | Stack | Summary |
|---|---------|-------|---------|
| 0 | [aicore](packages/aicore/) | Python | Shared platform layer: LLM (Large Language Model) clients, retries, tracing, cost, caching, guardrails, evals |
| 1 | [DocuMind](projects/01-documind/) | Flask, Postgres + pgvector | RAG (Retrieval-Augmented Generation) over documents: hybrid retrieval and reranking, cited answers, refusals, guardrails, HTTP (HyperText Transfer Protocol) API (Application Programming Interface) in Docker |
| 2 | [Deep Research](projects/02-deep-research/) | FastAPI + LangGraph | Multi-agent system: planner, parallel researchers, writer and critic produce a cited report (M2: LangGraph agent graph with critic and budgets) |
| 3 | [Text-to-SQL Agent](projects/03-text-to-sql-agent/) | Flask + LangGraph | Conversational chatbot over a SQL (Structured Query Language) database: multi-turn memory, read-only queries, self-correction |

## Quickstart

```bash
uv sync --all-packages                    # install all workspace members
docker compose -f infra/docker-compose.yml up -d   # Postgres+pgvector, Redis, Langfuse
uv run pytest                             # run tests
```
