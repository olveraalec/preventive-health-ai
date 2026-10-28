
FROM python:3.12-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.26

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src/ ./src/
RUN uv sync --frozen


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src

USER appuser

EXPOSE 8000

CMD ["uvicorn", "healthcare_ai_lab.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
