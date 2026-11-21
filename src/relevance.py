"""Is a ticker-tagged headline actually about that company?

FNSPID tags headlines with a ticker, but about a fifth of the sampled "TSLA" headlines are
about other companies, crypto or personal finance. This is a transparent keyword rule: a
headline is relevant if it mentions the ticker symbol, the company's name, or a strongly
associated term (product, executive). It is evaluated against the annotated headlines
(``python -m src.pipeline evaluate-relevance``).
"""

import re
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from src.company_names import AMBIGUOUS_SYMBOLS, COMPANY_TERMS, WEAK_TERM_EXCLUSIONS, WEAK_TERMS


def company_terms(ticker: str) -> List[str]:
    """Terms that make a headline about ``ticker``; always includes the symbol itself."""
    ticker = ticker.strip().upper()
    return [ticker, *COMPANY_TERMS.get(ticker, [])]


def _compile(terms: List[str]) -> re.Pattern:
    terms = sorted(terms, key=len, reverse=True)
    parts = []
    for term in terms:
        escaped = re.escape(term)
        if term in AMBIGUOUS_SYMBOLS:
            # symbol that is also a word: only "$ON", "(ON)", "NASDAQ:ON"
            parts.append(rf"(?:\$|\(|:\s?){escaped}(?![A-Za-z])")
        elif term.isupper() and len(term) <= 5:
            # ticker-like terms: case-sensitive, standalone or $-prefixed / in parentheses
            parts.append(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])")
        else:
            parts.append(rf"(?i:\b{escaped}(?:'s|’s|s)?\b)")
    return re.compile("|".join(parts)) if parts else re.compile(r"(?!x)x")  # never matches


@lru_cache(maxsize=None)
def _patterns(ticker: str):
    return (_compile(company_terms(ticker)),
            _compile(WEAK_TERMS.get(ticker, [])),
            _compile(WEAK_TERM_EXCLUSIONS.get(ticker, [])))


def is_relevant(title: str, ticker: str) -> bool:
    """Strong term (symbol, name, product) -> relevant. Weak term (e.g. the CEO's name) -> relevant
    only if no exclusion term (the CEO's other companies) appears."""
    strong, weak, exclude = _patterns(ticker.strip().upper())
    title = str(title)
    if strong.search(title):
        return True
    return bool(weak.search(title)) and not exclude.search(title)


def add_relevance(news: pd.DataFrame, title_col: str = "Article_title", ticker_col: str = "Stock_symbol",
                  ticker: str = None) -> pd.DataFrame:
    """Add a boolean ``relevant`` column (uses ``ticker`` for every row if given)."""
    out = news.copy()
    tickers = [ticker] * len(out) if ticker else out[ticker_col].astype(str).tolist()
    out["relevant"] = [is_relevant(t, k) for t, k in zip(out[title_col].astype(str), tickers)]
    return out


def relevance_metrics(predicted: List[bool], actual: List[bool]) -> Dict[str, float]:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(a and not p for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(actual), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / len(actual), 4) if actual else 0.0}
