import pandas as pd
import pytest

from src.relevance import add_relevance, company_terms, is_relevant, relevance_metrics


@pytest.mark.parametrize("title, expected", [
    ("Tesla Recalls 120,000 Vehicles", True),
    ("Why TSLA Stock Popped Today", True),
    ("Buying $TSLA ahead of earnings", True),
    ("Analysts weigh in on Tesla's margins", True),          # possessive
    ("Cybertruck orders top 1 million", True),               # product term
    ("Elon Musk says demand is strong", True),               # weak term, no exclusion
    ("Twitter hires law firm to sue Musk", False),           # weak term + other venture
    ("Elon Musk's SpaceX Capsule Links Up With Space Station", False),
    ("Is Lucid Group Stock a Buy Now?", False),
    ("5 Stocks That Moved the Market in April", False),
    ("Tsla-like returns are rare", False),                   # ticker match is case-sensitive
    ("STSLAX fund rebalances", False),                       # ticker must stand alone
])
def test_is_relevant_tesla(title, expected):
    assert is_relevant(title, "TSLA") is expected


def test_strong_term_wins_over_exclusion():
    assert is_relevant("Musk says Twitter deal won't distract from Tesla", "tsla")


def test_unknown_ticker_falls_back_to_symbol():
    assert company_terms("zzzz") == ["ZZZZ"]
    assert is_relevant("ZZZZ shares jump", "ZZZZ")
    assert not is_relevant("Shares jump", "ZZZZ")


def test_add_relevance_per_row_ticker():
    df = pd.DataFrame({"Article_title": ["Tesla up", "Tesla up"], "Stock_symbol": ["TSLA", "ZZZZ"]})
    assert add_relevance(df)["relevant"].tolist() == [True, False]
    assert add_relevance(df, ticker="TSLA")["relevant"].tolist() == [True, True]


def test_relevance_metrics():
    m = relevance_metrics([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5


@pytest.mark.parametrize("title, ticker, expected", [
    ("ON Semiconductor beats estimates", "ON", True),
    ("Buy $ON before earnings", "ON", True),
    ("onsemi (ON) raises guidance", "ON", True),
    ("US STOCKS-Wall St gains ON hopes of rate cuts", "ON", False),   # word, not the symbol
    ("FANG stocks lead Nasdaq higher", "FANG", False),
    ("Diamondback Energy (FANG) boosts dividend", "FANG", True),
    ("Pre-Market Most Active for Mar 3, 2023", "MAR", False),
    ("Marriott's RevPAR recovers", "MAR", True),
    ("Google faces antitrust trial", "GOOGL", True),
    ("Nvidia Is the AI Winner", "NVDA", True),
    ("Zoom Video tops estimates", "ZM", True),
    ("Stocks zoom higher on jobs data", "ZM", False),
])
def test_is_relevant_other_tickers(title, ticker, expected):
    assert is_relevant(title, ticker) is expected
