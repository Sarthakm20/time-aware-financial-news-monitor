from typing import TYPE_CHECKING, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_TO_ID,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
)

if TYPE_CHECKING:  # heavy imports only needed for type hints
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def clean_phrasebank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing rows, normalise whitespace/label case and drop exact duplicates."""
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
    id_col: str = "label_id",
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[id_col] = df_copy[label_col].map(LABEL_TO_ID)
    return df_copy


def prepare_model_dataframe(
    df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "label",
) -> pd.DataFrame:
    """Make a cleaned PhraseBank frame model-ready.

    Removes every sentence that appears with conflicting labels, keeps the first
    copy of remaining repeated sentences (so no text can leak across splits) and
    adds integer ``label_id``s.
    """
    label_counts = df.groupby(text_col)[label_col].nunique()
    conflicting = label_counts[label_counts > 1].index

    model_df = df[~df[text_col].isin(conflicting)]
    model_df = model_df.drop_duplicates(subset=text_col, keep="first").reset_index(drop=True)
    return add_label_ids(model_df, label_col=label_col)


def split_data_stratified(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = TRAIN_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

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
    label_col: str = "label_id",
) -> "Dataset":
    from datasets import Dataset

    subset = df[[text_col, label_col]].rename(columns={label_col: "labels"})
    return Dataset.from_pandas(subset, preserve_index=False)


def tokenize_hf_dataset(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_col: str = "sentence",
    max_length: int = MAX_SEQUENCE_LENGTH,
    batched: bool = True,
) -> "Dataset":
    def _tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize_fn, batched=batched)
