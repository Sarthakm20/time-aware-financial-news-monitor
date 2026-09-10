from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.config import (
    BASE_MODEL_NAME,
    DEFAULT_FINETUNED_MODEL_DIR,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_TRAIN_EPOCHS,
    WEIGHT_DECAY,
    ID_TO_LABEL,
    LABEL_TO_ID,
    METRICS_DIR,
)
from src.preprocessing import dataframe_to_hf_dataset, tokenize_hf_dataset
from src.evaluate import compute_trainer_metrics, save_metrics_json


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
) -> Dict[str, Any]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = dataframe_to_hf_dataset(train_df)
    val_dataset = dataframe_to_hf_dataset(val_df)

    tokenized_train = tokenize_hf_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_hf_dataset(val_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_trainer_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_payload = {
        "model": base_model_name,
        "training_examples": len(train_df),
        "validation_examples": len(val_df),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "validation_accuracy": eval_results.get("eval_accuracy"),
        "validation_macro_f1": eval_results.get("eval_macro_f1"),
    }
    save_metrics_json(metrics_payload, METRICS_DIR / "finetuned_validation_metrics.json")

    return metrics_payload

