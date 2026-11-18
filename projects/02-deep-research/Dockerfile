# Deep Research image: one image, two processes (see infra/docker-compose.yml):
#   API:    uvicorn (this CMD)
#   worker: `deep-research worker`
# Build context is the repository root (the uv workspace), because
# Deep Research depends on packages/aicore.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
COPY packages/aicore/pyproject.toml packages/aicore/README.md packages/aicore/
COPY projects/02-deep-research/pyproject.toml projects/02-deep-research/README.md projects/02-deep-research/
RUN uv sync --frozen --no-dev --package deep-research --no-install-workspace

# Then the code and the frozen corpus (data/corpus) the worker searches.
COPY packages/aicore packages/aicore
COPY projects/02-deep-research projects/02-deep-research
RUN uv sync --frozen --no-dev --package deep-research

RUN useradd --create-home app
USER app

EXPOSE 8000
# One process per container; scale with more containers. The API only
# talks to Redis and SSE streams are async, so one event loop serves
# many open streams.
CMD ["uvicorn", "--factory", "deep_research.api:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
