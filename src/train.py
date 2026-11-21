import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    BASE_MODEL_NAME,
    DEFAULT_FINETUNED_MODEL_DIR,
    EVAL_BATCH_SIZE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_TRAIN_EPOCHS,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)
from src.evaluate import calculate_metrics, compute_trainer_metrics
from src.preprocessing import add_label_ids, dataframe_to_hf_dataset, tokenize_hf_dataset


def train_distilbert_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path = DEFAULT_FINETUNED_MODEL_DIR,
    base_model_name: str = BASE_MODEL_NAME,
    num_epochs: int = NUM_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = RANDOM_SEED,
    class_weight_tensor: Optional[torch.Tensor] = None,
    keep_checkpoints: bool = True,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> Dict[str, Any]:
    """Fine-tune a sequence classifier and save the best (by validation macro F1) model.

    Pass ``class_weight_tensor`` to train with class-weighted cross-entropy.
    ``keep_checkpoints=False`` deletes the per-epoch checkpoints (~800 MB each) after saving.
    Returns validation metrics; writing them to disk is left to the caller.
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "label_id" not in train_df.columns:
        train_df = add_label_ids(train_df)
    if "label_id" not in val_df.columns:
        val_df = add_label_ids(val_df)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    tokenized_train = tokenize_hf_dataset(dataframe_to_hf_dataset(train_df), tokenizer, max_length=max_length)
    tokenized_val = tokenize_hf_dataset(dataframe_to_hf_dataset(val_df), tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=seed,
        report_to="none",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_trainer_metrics,
    )
    if class_weight_tensor is not None:
        from src.weighted_trainer import WeightedLossTrainer

        trainer = WeightedLossTrainer(class_weight_tensor=class_weight_tensor, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if not keep_checkpoints:
        shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)

    val_logits = trainer.predict(tokenized_val).predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    val_pred = [ID_TO_LABEL[int(i)] for i in np.argmax(val_logits, axis=-1)]
    val_metrics = calculate_metrics(val_df["label"].tolist(), val_pred)

    return {
        "model": base_model_name,
        "class_weighted_loss": class_weight_tensor is not None,
        "training_examples": int(len(train_df)),
        "validation_examples": int(len(val_df)),
        "epochs": num_epochs,
        "best_epoch": _best_epoch(trainer),
        "learning_rate": learning_rate,
        "validation_accuracy": val_metrics["accuracy"],
        "validation_macro_f1": val_metrics["macro_f1"],
        "per_class_validation_f1": val_metrics["per_class_f1"],
    }


def _best_epoch(trainer: Trainer) -> Optional[float]:
    best_metric = trainer.state.best_metric
    for entry in trainer.state.log_history:
        if entry.get("eval_macro_f1") == best_metric:
            return entry.get("epoch")
    return None
