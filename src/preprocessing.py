from typing import Tuple
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from src.config import (
    RANDOM_SEED,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
    LABEL_TO_ID,
    MAX_SEQUENCE_LENGTH,
)


def clean_phrasebank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(subset=["sentence", "label"]).copy()
    cleaned["sentence"] = (
        cleaned["sentence"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def add_label_ids(
    df: pd.DataFrame,
    label_col: str = "label",
    id_col: str = "label_id"
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[id_col] = df_copy[label_col].map(LABEL_TO_ID)
    return df_copy


def split_data_stratified(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = TRAIN_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-5, f"Split ratios must sum to 1.0, got {total}"

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[label_col],
        random_state=seed,
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_col],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def dataframe_to_hf_dataset(
    df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "label_id"
) -> Dataset:
    subset = df[[text_col, label_col]].rename(columns={label_col: "labels"})
    return Dataset.from_pandas(subset, preserve_index=False)


def tokenize_hf_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_col: str = "sentence",
    max_length: int = MAX_SEQUENCE_LENGTH,
    batched: bool = True,
) -> Dataset:
    def _tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize_fn, batched=batched)

