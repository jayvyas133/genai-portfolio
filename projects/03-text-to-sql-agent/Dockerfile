# Text-to-SQL chat API, served by gunicorn.
# Build context is the repository root (the uv workspace), because the
# project depends on packages/aicore. Built by infra/docker-compose.yml.
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
COPY projects/03-text-to-sql-agent/pyproject.toml projects/03-text-to-sql-agent/README.md projects/03-text-to-sql-agent/
RUN uv sync --frozen --no-dev --package text-to-sql-agent --no-install-workspace

# Then the code, the vendored Chinook script (for `load-db`) and the evals.
COPY packages/aicore packages/aicore
COPY projects/03-text-to-sql-agent projects/03-text-to-sql-agent
RUN uv sync --frozen --no-dev --package text-to-sql-agent

RUN useradd --create-home app
USER app

EXPOSE 8000
# 2 workers x 4 threads = 8 requests at once. Each worker has one event
# loop thread holding its pools, model client and agent graph.
# --timeout must exceed TEXT_TO_SQL_REQUEST_TIMEOUT_S (120 s) so the app
# answers 504 before gunicorn kills the worker.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", \
     "--timeout", "150", "--access-logfile", "-", "text_to_sql_agent.api:create_app()"]
