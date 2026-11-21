# Financial News Sentiment API (CPU)
#
#   docker build -t news-sentiment-api .
#
# Model weights are not baked into the image. Either point at the Hugging Face Hub:
#   docker run -p 8000:8000 -e SENTIMENT_MODEL=<hf-user>/distilbert-financial-headline-sentiment news-sentiment-api
# or mount a locally trained model:
#   docker run -p 8000:8000 -v "$PWD/models/distilbert_headline_adapted:/app/models/distilbert_headline_adapted:ro" \
#       news-sentiment-api
# Without a model, /health and the precomputed /tickers endpoints still work; /predict returns 503.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements-api.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

COPY src/ src/
COPY outputs/predictions/nasdaq100_scored_headlines_adapted.csv outputs/predictions/

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/models /app/.cache/huggingface \
 && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
