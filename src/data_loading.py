from pathlib import Path
from typing import Optional, List, Tuple
import zipfile
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PHRASEBANK_HF_REPO,
    PHRASEBANK_ZIP_FILE,
    PHRASEBANK_INNER_FILE,
    FNSPID_HF_DATASET,
    DEFAULT_TICKER,
)


def load_phrasebank_from_hub(
    repo_id: str = PHRASEBANK_HF_REPO,
    zip_filename: str = PHRASEBANK_ZIP_FILE,
    inner_file: str = PHRASEBANK_INNER_FILE,
) -> pd.DataFrame:
    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=zip_filename,
    )

    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for line in archive.read(inner_file).decode("latin-1").splitlines():
            if "@" in line:
                sentence, label = line.rsplit("@", 1)
                rows.append({
                    "sentence": sentence.strip(),
                    "label": label.strip().lower(),
                })

    return pd.DataFrame(rows)


def load_split_csvs(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, val_df, test_df


def stream_fnspid_slice(
    ticker: str = DEFAULT_TICKER,
    start_date: Optional[str] = "2019-01-01",
    end_date: Optional[str] = "2024-01-01",
    dataset_name: str = FNSPID_HF_DATASET,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    news_stream = load_dataset(dataset_name, split="train", streaming=True)

    start_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else None

    matched_rows: List[dict] = []
    ticker_upper = ticker.upper()

    for item in news_stream:
        stock_symbol = str(item.get("Stock_symbol", "")).upper()
        if stock_symbol != ticker_upper:
            continue

        raw_date = item.get("Date")
        if not raw_date:
            continue

        parsed_date = pd.to_datetime(raw_date, utc=True)
        if start_ts and parsed_date < start_ts:
            continue
        if end_ts and parsed_date >= end_ts:
            continue

        matched_rows.append({
            "Date": parsed_date,
            "Article_title": item.get("Article_title", "").strip(),
            "Stock_symbol": stock_symbol,
            "Url": item.get("Url", ""),
            "Publisher": item.get("Publisher", ""),
            "year": parsed_date.year,
        })

        if max_records and len(matched_rows) >= max_records:
            break

    df = pd.DataFrame(matched_rows)
    if not df.empty:
        df = df.sort_values("Date").drop_duplicates(subset=["Date", "Article_title"]).reset_index(drop=True)
    return df

