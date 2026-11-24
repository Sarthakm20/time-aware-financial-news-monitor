from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    DEFAULT_FINETUNED_MODEL_DIR,
    EVAL_BATCH_SIZE,
    ID_TO_LABEL,
    MAX_SEQUENCE_LENGTH,
)
from src.labels import label_column_order, summarize_probabilities

__all__ = ["HeadlineDataset", "label_column_order", "load_fine_tuned_model", "predict_sentiment_batch",
           "score_news_dataframe"]


class HeadlineDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: val[idx] for key, val in self.encodings.items()
        }


def load_fine_tuned_model(
    model_path: Union[str, Path] = DEFAULT_FINETUNED_MODEL_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path_str = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_str)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_sentiment_batch(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, class probabilities, confidences and P(pos) - P(neg) scores.

    Probability columns follow ``ID_TO_LABEL`` order: negative, neutral, positive,
    whatever order the model itself uses (see ``label_column_order``).
    """
    if len(texts) == 0:
        return summarize_probabilities(np.empty((0, len(ID_TO_LABEL)), dtype=np.float32))

    dataset = HeadlineDataset(list(texts), tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return summarize_probabilities(np.vstack(all_probs)[:, label_column_order(model)])


def score_news_dataframe(
    news_df: pd.DataFrame,
    text_col: str = "Article_title",
    model_path: Union[str, Path] = DEFAULT_FINETUNED_MODEL_DIR,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    model, tokenizer, device = load_fine_tuned_model(model_path)
    headlines = news_df[text_col].astype(str).tolist()

    labels, probs, confs, scores = predict_sentiment_batch(
        headlines,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    df_out = news_df.copy()
    df_out["predicted_label"] = labels
    df_out["negative_probability"] = probs[:, 0]
    df_out["neutral_probability"] = probs[:, 1]
    df_out["positive_probability"] = probs[:, 2]
    df_out["confidence"] = confs
    df_out["sentiment_score"] = scores

    return df_out

