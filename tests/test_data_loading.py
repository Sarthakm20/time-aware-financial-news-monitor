import pandas as pd

from src.data_loading import filter_news_records, parse_phrasebank_lines


def test_parse_phrasebank_lines_splits_on_last_at_sign():
    df = parse_phrasebank_lines([
        "Contact us @ the office for details .@neutral",
        "Profit rose 5 % .@Positive",
        "malformed line without label",
    ])
    assert df.to_dict("records") == [
        {"sentence": "Contact us @ the office for details .", "label": "neutral"},
        {"sentence": "Profit rose 5 % .", "label": "positive"},
    ]


def test_filter_news_records_applies_ticker_dates_and_cleaning():
    records = [
        {"Stock_symbol": " tsla ", "Date": "2020-05-01 10:00:00", "Article_title": " Tesla up ", "Url": "u1", "Publisher": "p"},
        {"Stock_symbol": "TSLA", "Date": "2020-05-01 10:00:00", "Article_title": "Tesla up", "Url": "u1", "Publisher": "p"},  # duplicate
        {"Stock_symbol": "AAPL", "Date": "2020-05-01", "Article_title": "Apple", "Url": "", "Publisher": ""},
        {"Stock_symbol": "TSLA", "Date": "not a date", "Article_title": "Bad date", "Url": "", "Publisher": ""},
        {"Stock_symbol": "TSLA", "Date": "2020-06-01", "Article_title": "   ", "Url": "", "Publisher": ""},
        {"Stock_symbol": "TSLA", "Date": "2018-12-31", "Article_title": "Too early", "Url": "", "Publisher": ""},
        {"Stock_symbol": "TSLA", "Date": "2024-01-01", "Article_title": "End is exclusive", "Url": "", "Publisher": ""},
        {"Stock_symbol": "TSLA", "Date": "2019-03-01", "Article_title": "Earlier", "Url": "u0", "Publisher": None},
    ]
    df = filter_news_records(records, ticker="TSLA", start_date="2019-01-01", end_date="2024-01-01")

    assert df["Article_title"].tolist() == ["Earlier", "Tesla up"]  # sorted by date
    assert df["Stock_symbol"].unique().tolist() == ["TSLA"]
    assert df["year"].tolist() == [2019, 2020]
    assert str(df["Date"].dt.tz) == "UTC"


def test_filter_news_records_respects_max_records_and_empty_input():
    records = [
        {"Stock_symbol": "TSLA", "Date": f"2020-01-0{i}", "Article_title": f"h{i}"} for i in range(1, 6)
    ]
    assert len(filter_news_records(records, max_records=2)) == 2

    empty = filter_news_records([], ticker="TSLA")
    assert empty.empty
    assert list(empty.columns) == ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "year"]
    assert isinstance(empty, pd.DataFrame)
