import pandas as pd
import pytest

from src import pipeline


def test_list_stages(capsys):
    assert pipeline.main(["--list"]) == 0
    out = capsys.readouterr().out
    for stage in pipeline.STAGES:
        assert stage in out


def test_resolve_stages():
    assert pipeline.resolve_stages(["all"]) == pipeline.ALL_STAGES
    assert set(pipeline.ALL_STAGES) <= set(pipeline.STAGES)
    assert "evaluate-headlines" not in pipeline.ALL_STAGES  # needs human labels
    assert pipeline.resolve_stages(["prepare", "monitor"]) == ["prepare", "monitor"]
    with pytest.raises(SystemExit):
        pipeline.resolve_stages(["nope"])


def test_prepare_and_monitor_end_to_end(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    rows = [{"sentence": f"{label} {i}", "label": label}
            for label in ["negative", "neutral", "positive"] for i in range(20)]
    pd.DataFrame(rows).to_csv(raw_dir / "financial_phrasebank_50agree_raw.csv", index=False)

    pred_dir = tmp_path / "outputs" / "predictions"
    pred_dir.mkdir(parents=True)
    pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=10, freq="7D", tz="UTC"),
        "sentiment_score": [0.5, -0.5] * 5,
        "predicted_label": ["positive", "negative"] * 5,
    }).to_csv(pred_dir / "tsla_scored_headlines_2019_2023.csv", index=False)

    args = ["prepare", "monitor", "--data-dir", str(tmp_path / "data"),
            "--outputs-dir", str(tmp_path / "outputs"), "--models-dir", str(tmp_path / "models")]
    assert pipeline.main(args) == 0

    processed = tmp_path / "data" / "processed"
    sizes = [len(pd.read_csv(processed / f"{n}.csv")) for n in ["train", "validation", "test"]]
    assert sizes == [42, 9, 9]
    summary = pd.read_csv(tmp_path / "outputs" / "metrics" / "tsla_sentiment_window_summary.csv")
    assert summary["window"].tolist()[-1] == "All available history"
    assert (tmp_path / "outputs" / "figures" / "tsla_30day_rolling_sentiment.png").exists()


def test_missing_input_gives_a_clear_error(tmp_path):
    args = ["monitor", "--data-dir", str(tmp_path / "d"), "--outputs-dir", str(tmp_path / "o"),
            "--models-dir", str(tmp_path / "m")]
    with pytest.raises(FileNotFoundError, match="run the 'score' stage first"):
        pipeline.main(args)


def test_model_slug():
    assert pipeline.model_slug("ProsusAI/finbert") == "finbert"
    assert pipeline.model_slug("microsoft/deberta-v3.small/") == "deberta-v3-small"


def _write_predictions(path, sentences, truth, predicted, confidence=0.9):
    rows = []
    for s, t, p in zip(sentences, truth, predicted):
        probs = {f"{lab}_probability": (confidence if lab == p else (1 - confidence) / 2)
                 for lab in ["negative", "neutral", "positive"]}
        rows.append({"sentence": s, "label": t, "predicted_label": p, **probs})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_analyze_stage_on_synthetic_predictions(tmp_path):
    labels = ["negative", "neutral", "positive"] * 20
    sentences = [f"s{i}" for i in range(len(labels))]
    processed = tmp_path / "data" / "processed"
    preds = tmp_path / "outputs" / "predictions"
    processed.mkdir(parents=True)
    preds.mkdir(parents=True)
    pd.DataFrame({"sentence": sentences, "label": labels}).to_csv(processed / "test.csv", index=False)

    good = [lab if i % 10 else "neutral" for i, lab in enumerate(labels)]  # 90%+ right
    _write_predictions(preds / "finetuned_test_predictions.csv", sentences, labels, good)
    _write_predictions(preds / "zero_shot_test_predictions.csv", sentences, labels, ["neutral"] * len(labels), 0.5)
    _write_predictions(preds / "finetuned_validation_predictions.csv", sentences, labels, good, 0.99)

    args = ["analyze", "--n-boot", "200", "--data-dir", str(tmp_path / "data"),
            "--outputs-dir", str(tmp_path / "outputs"), "--models-dir", str(tmp_path / "m")]
    assert pipeline.main(args) == 0

    table = pd.read_csv(tmp_path / "outputs" / "metrics" / "test_set_model_comparison.csv").set_index("model")
    assert table.loc["DistilBERT fine-tuned", "accuracy"] > table.loc["Zero-shot BART-MNLI", "accuracy"]
    assert table.loc["Zero-shot BART-MNLI", "mcnemar_p_value"] < 0.05
    assert "Majority class (negative)" in table.index or any(i.startswith("Majority") for i in table.index)
    import json
    report = json.loads((tmp_path / "outputs" / "metrics" / "test_set_analysis.json").read_text())
    assert report["calibration"]["temperature"] > 0
    assert (tmp_path / "outputs" / "figures" / "finetuned_reliability_diagram.png").exists()


def test_headline_sampling_and_evaluation_end_to_end(tmp_path):
    from src.labelling import gold_path, load_gold

    processed = tmp_path / "data" / "processed"
    preds = tmp_path / "outputs" / "predictions"
    processed.mkdir(parents=True)
    preds.mkdir(parents=True)
    dates = pd.date_range("2021-01-01", periods=40, freq="5D", tz="UTC")
    titles = [f"Tesla headline {i}" for i in range(40)]
    pd.DataFrame({"Date": dates, "Article_title": titles, "Stock_symbol": "TSLA"}).to_csv(
        processed / "fnspid_tsla_2019_2023.csv", index=False)
    model_labels = ["positive", "neutral", "negative", "neutral"] * 10
    pd.DataFrame({"Date": dates, "Article_title": titles, "predicted_label": model_labels}).to_csv(
        preds / "tsla_scored_headlines_2019_2023.csv", index=False)

    common = ["--data-dir", str(tmp_path / "data"), "--outputs-dir", str(tmp_path / "outputs"),
              "--models-dir", str(tmp_path / "m"), "--n-boot", "200"]
    assert pipeline.main(["sample-headlines", "--n-label", "20", *common]) == 0
    gold_file = gold_path("TSLA", tmp_path / "data" / "labelled")
    gold = load_gold(gold_file)
    assert len(gold) == 20

    with pytest.raises(SystemExit, match="No labelled headlines"):
        pipeline.main(["evaluate-headlines", *common])

    # "annotate": agree with the model except on the first two items
    truth = dict(zip(titles, model_labels))
    gold["label"] = [truth[t] for t in gold["Article_title"]]
    gold.loc[0, "label"] = "skip"
    gold.loc[1, "label"] = "positive" if gold.loc[1, "label"] != "positive" else "negative"
    gold.to_csv(gold_file, index=False)

    assert pipeline.main(["sample-headlines", "--n-label", "20", *common]) == 0  # must not overwrite labels
    assert load_gold(gold_file)["label"].iloc[0] == "skip"

    assert pipeline.main(["evaluate-headlines", *common]) == 0
    import json
    report = json.loads((tmp_path / "outputs" / "metrics" / "tsla_headline_evaluation.json").read_text())
    assert report["labelled"] == 19 and report["skipped"] == 1
    distil = next(m for m in report["models"] if m["model"] == "DistilBERT fine-tuned")
    assert distil["accuracy"] == pytest.approx(18 / 19, abs=1e-4)


@pytest.mark.parametrize("title, expected", [
    ("US STOCKS-Wall St closes up on hopes of abating Fed", True),
    ("Dow Reverses Losses, Settles Modestly Higher", True),
    ("Dow, Nasdaq Pivot Higher to Score Triple-Digit Gains", True),
    ("Pre-Market Most Active for Oct 12, 2022 : TQQQ, SQQQ, TSLA", True),
    ("Downgrade hits Tesla shares", False),           # word boundary: "Down..." is not "Dow"
    ("Tesla Shares Slide After Delivery Miss", False),
    ("Why Wall Street Loves Tesla", False),           # anchored at the start
])
def test_market_wide_headline_pattern(title, expected):
    import re

    from src.config import MARKET_WIDE_HEADLINE_PATTERN

    assert bool(re.search(MARKET_WIDE_HEADLINE_PATTERN, title, flags=re.I)) is expected


def test_summarize_seed_runs():
    runs = pd.DataFrame({"seed": [1, 2, 3], "source": ["trained"] * 3,
                         "accuracy": [0.80, 0.82, 0.84], "macro_f1": [0.7, 0.7, 0.7]})
    summary = pipeline.summarize_seed_runs(runs, epochs=2)
    assert summary["n_seeds"] == 3 and summary["seeds"] == [1, 2, 3]
    assert summary["accuracy"] == {"mean": 0.82, "std": 0.02, "min": 0.8, "max": 0.84}
    assert summary["macro_f1"]["std"] == 0.0


def test_event_study_stage_end_to_end(tmp_path):
    import json

    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", periods=260)
    rows, scored = [], []
    qqq = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    rows += [{"Date": d, "ticker": "QQQ", "close": c, "volume": 1e6} for d, c in zip(dates, qqq)]
    for t in ["AAA", "BBB", "CCC"]:
        sentiment = rng.uniform(-1, 1, len(dates))
        ar = 0.02 * np.roll(sentiment, 1) + rng.normal(0, 0.005, len(dates))  # day+1 reacts to sentiment
        stock_ret = np.r_[0, np.diff(qqq) / qqq[:-1]] + ar
        close = 100 * np.cumprod(1 + stock_ret)
        rows += [{"Date": d, "ticker": t, "close": c, "volume": 1e6} for d, c in zip(dates, close)]
        scored += [{"Date": f"{d.date()} 00:00:00+00:00", "Stock_symbol": t, "Article_title": f"{t} news {i}",
                    "sentiment_score": s} for i, (d, s) in enumerate(zip(dates, sentiment))]
    (tmp_path / "data" / "prices").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "prices" / "daily.csv", index=False)
    (tmp_path / "outputs" / "predictions").mkdir(parents=True)
    pd.DataFrame(scored).to_csv(tmp_path / "outputs" / "predictions" / "nasdaq100_scored_headlines_base.csv",
                                index=False)

    args = ["event-study", "--n-event-boot", "100", "--data-dir", str(tmp_path / "data"),
            "--outputs-dir", str(tmp_path / "outputs"), "--models-dir", str(tmp_path / "m")]
    assert pipeline.main(args) == 0

    ic = pd.read_csv(tmp_path / "outputs" / "metrics" / "event_study_information_coefficients.csv")
    p1 = ic[(ic.outcome == "ar_p1") & (ic.headlines == "all relevant headlines")].iloc[0]
    assert p1["value"] > 0.5 and p1["ci_low"] > 0              # planted day+1 effect is found
    m1 = ic[(ic.outcome == "ar_m1") & (ic.headlines == "all relevant headlines")].iloc[0]
    assert m1["ci_low"] < 0 < m1["ci_high"]                    # no effect before the news
    summary = json.loads((tmp_path / "outputs" / "metrics" / "event_study_summary.json").read_text())
    assert summary["runs"][0]["tickers"] == 3
    assert (tmp_path / "outputs" / "figures" / "event_study_car_base.png").exists()
