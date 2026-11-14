# Runtime image for finint-agent. Data and secrets are mounted at run time, never baked in.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer first, so code changes don't reinstall everything.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group agent --group ui --no-install-project

COPY . ./
RUN uv sync --locked --no-dev --group agent --group ui

ENTRYPOINT ["uv", "run", "--no-sync", "finint"]
CMD ["--help"]
