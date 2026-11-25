# syntax=docker/dockerfile:1

# ---- base: Python + uv ----
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# ---- builder: runtime deps + project, no dev deps ----
FROM base AS builder
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- test: adds dev deps + tests, used by the agent-eval CI job ----
FROM builder AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
COPY tests ./tests
COPY scenarios ./scenarios
ENV PATH="/app/.venv/bin:$PATH" \
    EQUIP_DB_PATH=/app/data/equipment.db \
    AS_OF_DATE=2026-12-01
CMD ["pytest", "-m", "eval"]

# ---- runtime (default target): one image; `server` or the agent (see docker-compose.yml) ----
FROM base AS runtime
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown app:app /app/data
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app scenarios ./scenarios
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENV PATH="/app/.venv/bin:$PATH" \
    EQUIP_DB_PATH=/app/data/equipment.db
USER app
VOLUME ["/app/data"]
EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
