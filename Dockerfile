FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY demo ./demo
RUN pip install --no-cache-dir ".[vi]"

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=5
CMD ["sh", "-c", "exec uvicorn kg_generator.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
