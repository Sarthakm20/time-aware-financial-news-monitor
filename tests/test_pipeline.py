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
