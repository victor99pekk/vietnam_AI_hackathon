FROM python:3.12-slim

ARG TARGETARCH
WORKDIR /app
COPY pyproject.toml ./
COPY README.md ./
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=6
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      pip install --no-cache-dir "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir "torch==2.6.0"; \
    fi
RUN pip install --no-cache-dir -e ".[all]"
RUN python -m spacy download en_core_web_sm
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
COPY src ./src
COPY project_pitch/demo ./demo
# Sample KG data — provide via build arg or volume mount when available
# COPY data/samples/sample_kg.json ./data/global_sample.json

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
CMD ["sh", "-c", "exec uvicorn kg_generator.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
