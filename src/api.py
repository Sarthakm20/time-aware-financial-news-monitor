"""REST API for headline sentiment.

Run locally:   uvicorn src.api:app --port 8000      (docs at http://localhost:8000/docs)
Docker:        docker build -t news-sentiment-api . && docker run -p 8000:8000 news-sentiment-api

Endpoints
---------
GET  /health                              service status and model name
POST /predict                             score up to 256 headlines (optionally with relevance for a ticker)
GET  /tickers                             tickers with precomputed scores
GET  /tickers/{symbol}/sentiment          lookback-window summary + latest headlines
GET  /tickers/{symbol}/timeline           rolling sentiment series (weekly points)

Sentiment summarises the tone of headlines. It is not a trading signal or investment advice.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import LABELS
from src.explorer import (
    DEFAULT_SCORED_CSV,
    load_scored,
    ticker_headlines,
    ticker_overview,
    ticker_timeline,
    ticker_windows,
)
from src.service import MAX_HEADLINE_CHARS, MAX_HEADLINES_PER_REQUEST, SentimentService

DISCLAIMER = "Summarises headline tone; not a trading signal or investment advice."


class PredictRequest(BaseModel):
    headlines: List[str] = Field(..., min_length=1, max_length=MAX_HEADLINES_PER_REQUEST,
                                 examples=[["Tesla deliveries beat estimates", "Intel cuts guidance"]])
    ticker: Optional[str] = Field(None, max_length=10, description="If set, also report whether each "
                                  "headline is about this company (relevance filter).")


class PredictionOut(BaseModel):
    headline: str
    label: str
    probabilities: dict
    sentiment_score: float = Field(..., description="P(positive) - P(negative), in [-1, 1]")
    confidence: float
    relevant: Optional[bool] = None


class PredictResponse(BaseModel):
    model: str
    labels: List[str]
    predictions: List[PredictionOut]
    disclaimer: str = DISCLAIMER


def create_app(service: Optional[SentimentService] = None, scored_csv: Path = DEFAULT_SCORED_CSV) -> FastAPI:
    service = service or SentimentService()
    app = FastAPI(
        title="Financial News Sentiment API",
        version="1.0.0",
        description="Headline sentiment from a DistilBERT model fine-tuned on Financial PhraseBank and "
                    "adapted to news headlines, plus precomputed Nasdaq-100 sentiment (2019-2023). "
                    + DISCLAIMER,
    )

    def scored():
        try:
            return load_scored(scored_csv)
        except FileNotFoundError:
            raise HTTPException(503, "precomputed scores not available on this server") from None

    def ticker_or_404(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, f"unknown ticker {args[1]!r}") from None

    @app.get("/health")
    def health():
        try:
            name = service.model_name
        except FileNotFoundError:
            name = None
        return {"status": "ok", "model": name, "model_loaded": service.loaded}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if any(len(h) > MAX_HEADLINE_CHARS for h in req.headlines):
            raise HTTPException(422, f"headlines must be at most {MAX_HEADLINE_CHARS} characters")
        try:
            preds = service.predict(req.headlines, ticker=req.ticker)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except OSError:  # no local model, or the Hub model could not be downloaded
            raise HTTPException(503, "sentiment model not available; set SENTIMENT_MODEL") from None
        return {"model": service.model_name, "labels": LABELS, "predictions": [p.to_dict() for p in preds]}

    @app.get("/tickers")
    def tickers():
        return {"tickers": ticker_overview(scored()).to_dict(orient="records")}

    @app.get("/tickers/{symbol}/sentiment")
    def ticker_sentiment(symbol: str, half_life_days: float = Query(30, gt=0, le=3650),
                         n_headlines: int = Query(10, ge=0, le=100)):
        data = scored()
        windows = ticker_or_404(ticker_windows, data, symbol, half_life_days=half_life_days)
        return {
            "ticker": symbol.upper(),
            "half_life_days": half_life_days,
            "windows": windows.to_dict(orient="records"),
            "latest_headlines": ticker_headlines(data, symbol, n=n_headlines),
            "disclaimer": DISCLAIMER,
        }

    @app.get("/tickers/{symbol}/timeline")
    def ticker_timeline_endpoint(symbol: str, window_days: int = Query(30, ge=1, le=365)):
        tl = ticker_or_404(ticker_timeline, scored(), symbol, window_days=window_days)
        tl["Date"] = tl["Date"].dt.date.astype(str)
        return {"ticker": symbol.upper(), "window_days": window_days,
                "points": tl.rename(columns={"Date": "date"}).to_dict(orient="records")}

    return app


app = create_app()
