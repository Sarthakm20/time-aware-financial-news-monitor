"""JSON endpoints behind the dashboard site (all read-only, all under /api).

Numbers come from the scored headlines (filterable by company, year and model) or from the committed
evaluation/event-study files (always all companies, 2019-2023). See docs/superpowers/specs/ for the design.
"""

from pathlib import Path
from typing import Callable, Dict, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.dashboard_data import (
    YEARS,
    clean,
    company_view,
    filter_scored,
    search_headlines,
    sentiment_distribution,
    trend_points,
)
from src.dashboard_live import fetch_close, live_summary
from src.dashboard_metrics import (
    MetricsUnavailable,
    corpus,
    headline_f1,
    market_impact,
    market_relationship,
    model_evaluation,
    relevance,
    scope_label,
)
from src.explorer import load_scored, ticker_overview

DISCLAIMER = "Summarises headline tone; not a trading signal or investment advice."
MODELS = [{"id": "adapted", "label": "Headline-adapted DistilBERT"}, {"id": "base", "label": "PhraseBank DistilBERT"}]

Model = Literal["adapted", "base"]
Period = Optional[int]


def build_dashboard_router(scored_paths: Dict[str, Path], metrics_dir: Path, live_db: Path,
                           price_fetcher: Callable = fetch_close) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["dashboard"])

    def scored(model: str) -> pd.DataFrame:
        try:
            return load_scored(Path(scored_paths[model]))
        except FileNotFoundError:
            raise HTTPException(503, "precomputed scores not available on this server") from None

    def guarded(fn: Callable, *args, **kwargs):
        """Run a payload builder, mapping missing files and unknown tickers to clear HTTP errors."""
        try:
            return fn(*args, **kwargs)
        except MetricsUnavailable as exc:
            raise HTTPException(503, f"{exc.args[0]} not available on this server") from None
        except KeyError as exc:
            raise HTTPException(404, f"No company called {str(exc.args[0]).upper()} in the dataset.") from None

    def period_query():
        return Query(None, ge=min(YEARS), le=max(YEARS), description="Calendar year; omit for all years")

    @router.get("/meta")
    def meta():
        overview = ticker_overview(scored("adapted"))
        return {"tickers": overview[["ticker", "headlines", "first_date", "last_date"]].to_dict(orient="records"),
                "years": list(YEARS), "models": MODELS, "corpus": corpus(metrics_dir), "disclaimer": DISCLAIMER}

    @router.get("/overview")
    def overview(ticker: Optional[str] = None, period: Period = period_query(), model: Model = "adapted"):
        data = scored(model)

        def build():
            sub = filter_scored(data, ticker, period)
            if ticker:
                company = filter_scored(data, ticker)
                coverage = {"kind": "company", "ticker": ticker.upper(),
                            "first_date": str(company["Date"].min().date()),
                            "last_date": str(company["Date"].max().date())}
            else:
                coverage = {"kind": "universe", "companies": int(data["Stock_symbol"].nunique())}
            c = corpus(metrics_dir) or {}
            return clean({
                "scope": {"ticker": ticker.upper() if ticker else None, "period": period, "model": model},
                "cards": {
                    "coverage": coverage,
                    "headlines": {"scored": int(len(sub)),
                                  "collected": c.get("collected_headlines") if not (ticker or period) else None},
                    "f1": headline_f1(metrics_dir, model, ticker),
                    "company_days": {"value": market_impact(metrics_dir, model, "all", period)["events"],
                                     "scope": scope_label(period)},
                },
                "distribution": sentiment_distribution(sub),
                "mean_sentiment": float(sub["sentiment_score"].mean()) if len(sub) else None,
                "trend": trend_points(sub),
                "relevance": relevance(metrics_dir),
                "market": market_relationship(metrics_dir, model, "excluding", period),
                "disclaimer": DISCLAIMER,
            })

        return guarded(build)

    @router.get("/company/{ticker}")
    def company(ticker: str, period: Period = period_query(), model: Model = "adapted",
                window_days: int = Query(30, ge=7, le=90), half_life_days: float = Query(30, ge=7, le=180),
                order: Literal["latest", "most_positive", "most_negative"] = "latest",
                n: int = Query(12, ge=1, le=100)):
        data = scored(model)

        def build():
            view = company_view(data, ticker, year=period, window_days=window_days, half_life_days=half_life_days,
                                order=order, n=n)
            f1 = headline_f1(metrics_dir, model, ticker)
            annotated = sorted({row["ticker"] for row in model_evaluation(metrics_dir)["per_company"]})
            has_rows = view["cards"]["headlines"] > 0
            return {**view, "accuracy": f1 if f1["scope"] == view["ticker"] and has_rows else None,
                    "accuracy_available_for": annotated, "disclaimer": DISCLAIMER}

        return guarded(build)

    @router.get("/headlines")
    def headlines(q: Optional[str] = Query(None, max_length=200), ticker: Optional[str] = None,
                  label: Optional[Literal["positive", "neutral", "negative"]] = None,
                  start: Optional[str] = None, end: Optional[str] = None,
                  sort: Literal["newest", "oldest", "most_positive", "most_negative"] = "newest",
                  page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100), model: Model = "adapted"):
        for value in (start, end):
            if value is not None:
                try:
                    pd.Timestamp(value)
                except ValueError:
                    raise HTTPException(422, f"not a date: {value!r} (use YYYY-MM-DD)") from None
        return search_headlines(scored(model), q=q, ticker=ticker, label=label, start=start, end=end, sort=sort,
                                page=page, page_size=page_size)

    @router.get("/market-impact")
    def market(model: Model = "adapted", headlines: Literal["excluding", "all"] = "excluding",
               period: Period = period_query()):
        return guarded(market_impact, metrics_dir, model, headlines, period)

    @router.get("/model-evaluation")
    def models():
        return guarded(model_evaluation, metrics_dir)

    @router.get("/price/{ticker}")
    def price(ticker: str, start: str, end: str):
        return {"ticker": ticker.upper(), "points": price_fetcher(ticker.upper(), start, end)}

    @router.get("/live/summary")
    def live(ticker: Optional[str] = None):
        return live_summary(Path(live_db), ticker)

    return router
