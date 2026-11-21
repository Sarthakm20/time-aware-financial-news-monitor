"""The Hugging Face Space bundle must contain everything the dashboard imports and reads."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish", ROOT / "deploy" / "publish.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


@pytest.mark.skipif(not all((ROOT / f).exists() for f in publish.SPACE_FILES), reason="outputs not available")
def test_space_bundle_runs_the_dashboard_on_its_own(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    repo = publish.stage_space(tmp_path, "someone")
    assert repo == "someone/news-sentiment-monitor"
    for name in ["Dockerfile", "README.md", "requirements.txt"]:
        assert (tmp_path / name).exists()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme and "<hf-user>" not in readme

    # Run the dashboard from the bundle only: the project's own src/ must not be importable.
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path if Path(p).resolve() != ROOT])
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, mod)
    at = AppTest.from_file(str(tmp_path / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].label == "Headlines"


def test_model_card_has_license_and_no_placeholders_after_staging(tmp_path):
    card = (ROOT / "deploy" / "model_card.md").read_text(encoding="utf-8")
    assert "license: cc-by-nc-sa-3.0" in card  # inherited from Financial PhraseBank
    assert "not human-verified" in card
    if all((publish.MODEL_DIR / f).exists() for f in publish.MODEL_FILES):
        publish.stage_model(tmp_path, "someone")
        staged = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert "<hf-user>" not in staged and "someone/distilbert-financial-headline-sentiment" in staged
