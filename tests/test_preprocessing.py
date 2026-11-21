import pandas as pd
import pytest

from src.config import LABEL_TO_ID, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.preprocessing import (
    add_label_ids,
    clean_phrasebank_dataframe,
    prepare_model_dataframe,
    split_data_stratified,
)


def _balanced_frame(n_per_label: int = 40) -> pd.DataFrame:
    rows = []
    for label in ["negative", "neutral", "positive"]:
        for i in range(n_per_label):
            rows.append({"sentence": f"{label} sentence {i}", "label": label})
    return pd.DataFrame(rows)


def test_clean_normalises_whitespace_and_labels_and_drops_exact_duplicates():
    raw = pd.DataFrame({
        "sentence": ["  Sales   rose  ", "Sales rose", None, "Costs\tfell"],
        "label": ["Positive ", "positive", "neutral", "NEGATIVE"],
    })
    cleaned = clean_phrasebank_dataframe(raw)

    assert cleaned["sentence"].tolist() == ["Sales rose", "Costs fell"]
    assert cleaned["label"].tolist() == ["positive", "negative"]


def test_prepare_removes_conflicting_sentences_and_dedupes():
    df = pd.DataFrame({
        "sentence": ["a", "a", "b", "b", "c"],
        "label": ["positive", "negative", "neutral", "neutral", "positive"],
    })
    model_df = prepare_model_dataframe(df)

    # "a" has two different labels -> removed entirely; "b" repeated -> kept once
    assert model_df["sentence"].tolist() == ["b", "c"]
    assert model_df["label_id"].tolist() == [LABEL_TO_ID["neutral"], LABEL_TO_ID["positive"]]


def test_add_label_ids_maps_every_label():
    out = add_label_ids(pd.DataFrame({"label": ["negative", "neutral", "positive"]}))
    assert out["label_id"].tolist() == [0, 1, 2]


def test_split_sizes_are_disjoint_and_stratified():
    df = _balanced_frame(40)  # 120 rows
    train, val, test = split_data_stratified(df, seed=0)

    assert (len(train), len(val), len(test)) == (84, 18, 18)
    assert set(train["sentence"]).isdisjoint(val["sentence"])
    assert set(train["sentence"]).isdisjoint(test["sentence"])
    assert set(val["sentence"]).isdisjoint(test["sentence"])
    for split in (train, val, test):
        shares = split["label"].value_counts(normalize=True)
        assert shares.max() - shares.min() < 0.02  # each class ~1/3


def test_split_is_deterministic_for_a_seed():
    df = _balanced_frame(20)
    first = split_data_stratified(df, seed=7)
    second = split_data_stratified(df, seed=7)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        split_data_stratified(_balanced_frame(5), train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)


@pytest.mark.skipif(
    not (RAW_DATA_DIR / "financial_phrasebank_50agree_raw.csv").exists(),
    reason="committed PhraseBank data not available",
)
def test_committed_splits_are_reproducible_from_raw_data():
    """Regression guard: the published train/val/test CSVs must come out of src/ exactly."""
    raw = pd.read_csv(RAW_DATA_DIR / "financial_phrasebank_50agree_raw.csv")
    model_df = prepare_model_dataframe(clean_phrasebank_dataframe(raw))
    train, val, test = split_data_stratified(model_df)

    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        committed = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), committed, check_dtype=False)
