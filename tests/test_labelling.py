import pandas as pd
import pytest

from src.labelling import interactive_label, labelled_rows, load_gold, progress_line, sample_for_annotation


def _news(n_per_year=50):
    rows = []
    for year in [2020, 2021, 2022]:
        for i in range(n_per_year):
            rows.append({"Date": f"{year}-03-{(i % 28) + 1:02d} {i % 24:02d}:00:00+00:00",
                         "Article_title": f"headline {year}-{i}",
                         "predicted_label": "positive", "sentiment_score": 0.9})
    return pd.DataFrame(rows)


def test_sample_is_blind_sized_and_year_stratified():
    sample = sample_for_annotation(_news(), n=30, seed=1)
    assert len(sample) == 30
    assert list(sample.columns) == ["item_id", "Date", "Article_title", "label"]  # no model output leaks
    assert sample["item_id"].tolist() == list(range(1, 31))
    assert (sample["label"] == "").all()
    assert sample["Date"].dt.year.value_counts().to_dict() == {2020: 10, 2021: 10, 2022: 10}
    assert sample["Article_title"].is_unique


def test_sample_is_reproducible_and_capped():
    a = sample_for_annotation(_news(), n=20, seed=3)
    b = sample_for_annotation(_news(), n=20, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert len(sample_for_annotation(_news(5), n=1000)) == 15


def _gold_file(tmp_path, labels):
    df = pd.DataFrame({"item_id": range(1, len(labels) + 1), "Date": "2021-01-01",
                       "Article_title": [f"h{i}" for i in range(len(labels))], "label": labels})
    path = tmp_path / "gold.csv"
    df.to_csv(path, index=False)
    return path


def test_load_gold_normalises_and_validates(tmp_path):
    df = load_gold(_gold_file(tmp_path, [" Positive", "", "skip", "NEUTRAL"]))
    assert df["label"].tolist() == ["positive", "", "skip", "neutral"]
    assert labelled_rows(df)["label"].tolist() == ["positive", "neutral"]
    assert "3/4 done" in progress_line(df)  # skip counts as done

    with pytest.raises(ValueError, match="unknown labels"):
        load_gold(_gold_file(tmp_path, ["bullish"]))


def test_interactive_label_saves_supports_back_and_resume(tmp_path):
    path = _gold_file(tmp_path, ["", "", "", ""])
    keys = iter(["p", "x", "n", "b", "u", "q"])  # p; invalid; n then undo -> u; quit
    interactive_label(path, input_fn=lambda _: next(keys), print_fn=lambda *_: None)
    assert load_gold(path)["label"].tolist() == ["positive", "neutral", "", ""]

    keys = iter(["s", "n"])  # resume from the first unlabelled row
    interactive_label(path, input_fn=lambda _: next(keys), print_fn=lambda *_: None)
    assert load_gold(path)["label"].tolist() == ["positive", "neutral", "skip", "negative"]
