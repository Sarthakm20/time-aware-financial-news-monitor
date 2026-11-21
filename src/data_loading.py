import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TICKER,
    FNSPID_HF_DATASET,
    PHRASEBANK_HF_REPO,
    PHRASEBANK_INNER_FILE,
    PHRASEBANK_ZIP_FILE,
    PROCESSED_DATA_DIR,
)

ALL_TICKERS = {"ALL", "*"}


def parse_phrasebank_lines(lines: Iterable[str]) -> pd.DataFrame:
    """Parse ``sentence@label`` lines from the PhraseBank text files."""
    rows = []
    for line in lines:
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
    return pd.DataFrame(rows, columns=["sentence", "label"])


def load_phrasebank_from_hub(
    repo_id: str = PHRASEBANK_HF_REPO,
    zip_filename: str = PHRASEBANK_ZIP_FILE,
    inner_file: str = PHRASEBANK_INNER_FILE,
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=zip_filename,
    )
    with zipfile.ZipFile(zip_path) as archive:
        lines = archive.read(inner_file).decode("latin-1").splitlines()
    return parse_phrasebank_lines(lines)


def load_split_csvs(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, val_df, test_df


def filter_news_records(
    records: Iterable[dict],
    ticker: str = DEFAULT_TICKER,
    start_date: Optional[str] = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Keep one ticker's (or, with ``ticker="ALL"``, every ticker's) dated, non-empty headlines
    in ``[start_date, end_date)``.

    Works on any iterable of FNSPID-style dicts, so it can be tested without a download.
    """
    start_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else None
    ticker_upper = ticker.strip().upper()
    keep_all = ticker_upper in ALL_TICKERS

    matched_rows: List[dict] = []
    for item in records:
        stock_symbol = str(item.get("Stock_symbol") or "").strip().upper()
        if not stock_symbol or (not keep_all and stock_symbol != ticker_upper):
            continue

        parsed_date = pd.to_datetime(item.get("Date"), errors="coerce", utc=True)
        if pd.isna(parsed_date):
            continue
        if start_ts is not None and parsed_date < start_ts:
            continue
        if end_ts is not None and parsed_date >= end_ts:
            continue

        headline = str(item.get("Article_title") or "").strip()
        if not headline:
            continue

        matched_rows.append({
            "Date": parsed_date,
            "Article_title": headline,
            "Stock_symbol": stock_symbol,
            "Url": item.get("Url"),
            "Publisher": item.get("Publisher"),
        })

        if max_records and len(matched_rows) >= max_records:
            break

    columns = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "year"]
    df = pd.DataFrame(matched_rows, columns=columns[:-1])
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = (
        df.drop_duplicates(subset=["Date", "Article_title", "Stock_symbol"])
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    df["year"] = df["Date"].dt.year
    return df


def stream_fnspid_slice(
    ticker: str = DEFAULT_TICKER,
    start_date: Optional[str] = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    dataset_name: str = FNSPID_HF_DATASET,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Stream FNSPID from the Hugging Face Hub and return one ticker's headlines."""
    from datasets import load_dataset

    news_stream = load_dataset(dataset_name, split="train", streaming=True)
    return filter_news_records(
        news_stream,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        max_records=max_records,
    )
