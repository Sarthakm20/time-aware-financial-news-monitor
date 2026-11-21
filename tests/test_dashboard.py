"""Headless run of the Streamlit dashboard against the committed outputs/ files."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "outputs" / "predictions" / "nasdaq100_scored_headlines_adapted.csv"


@pytest.mark.skipif(not DATA.exists(), reason="precomputed Nasdaq-100 scores not available")
def test_dashboard_renders_and_switches_ticker_and_model():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["🔎 Company explorer", "📈 Market reaction", "🧪 Model quality",
                                         "✍️ Score a headline"]
    assert at.metric[0].label == "Headlines" and int(at.metric[0].value.replace(",", "")) > 0

    at.sidebar.selectbox[0].select("NVDA").run()
    at.sidebar.radio[0].set_value("PhraseBank DistilBERT").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Domain shift" in w.value for w in at.warning)  # the caveat is always shown
