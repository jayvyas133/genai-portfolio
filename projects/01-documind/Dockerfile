# DocuMind HTTP API, served by gunicorn.
# Build context is the repository root (the uv workspace), because
# DocuMind depends on packages/aicore. Built by infra/docker-compose.yml.
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
COPY projects/01-documind/pyproject.toml projects/01-documind/README.md projects/01-documind/
RUN uv sync --frozen --no-dev --package documind --no-install-workspace

# Then the code (and data/, so `documind ingest` also works in the container).
COPY packages/aicore packages/aicore
COPY projects/01-documind projects/01-documind
RUN uv sync --frozen --no-dev --package documind

RUN useradd --create-home app
USER app

EXPOSE 8000
# 2 workers x 4 threads = 8 requests at once; each thread runs its own
# event loop. --timeout must exceed DOCUMIND_REQUEST_TIMEOUT_S (30 s) so
# the app answers 504 before gunicorn kills the worker.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", \
     "--timeout", "60", "--access-logfile", "-", "documind.api:create_app()"]
