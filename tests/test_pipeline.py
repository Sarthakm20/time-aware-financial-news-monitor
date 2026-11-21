import pandas as pd
import pytest

from src import pipeline


def test_list_stages(capsys):
    assert pipeline.main(["--list"]) == 0
    out = capsys.readouterr().out
    for stage in pipeline.STAGES:
        assert stage in out


def test_resolve_stages():
    assert pipeline.resolve_stages(["all"]) == list(pipeline.STAGES)
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
