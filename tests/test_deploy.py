"""The Hugging Face Space bundle must contain everything the dashboard site imports and reads."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish", ROOT / "deploy" / "publish.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


@pytest.mark.skipif(not all((ROOT / f).exists() for f in publish.SPACE_FILES), reason="outputs not available")
def test_space_bundle_serves_the_site_on_its_own(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    repo = publish.stage_space(tmp_path, "someone")
    assert repo == "someone/news-sentiment-monitor"
    for name in ["Dockerfile", "README.md", "requirements.txt"]:
        assert (tmp_path / name).exists()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme and "app_port: 7860" in readme and "<hf-user>" not in readme
    assert "uvicorn" in (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert "streamlit" not in (tmp_path / "requirements.txt").read_text(encoding="utf-8").lower()
    assert not (tmp_path / "app").exists()

    # Serve the site from the bundle only: the project's own src/ must not be importable.
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path if Path(p).resolve() != ROOT])
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, mod)
    try:
        api = importlib.import_module("src.api")
        assert Path(api.__file__).resolve().is_relative_to(tmp_path.resolve())
        client = TestClient(api.create_app())  # every default path now resolves inside the bundle
        assert "Market Impact" in client.get("/").text
        assert client.get("/static/js/pages/overview.js").status_code == 200
        overview = client.get("/api/overview")
        assert overview.status_code == 200 and overview.json()["cards"]["company_days"]["value"] == 24807
        assert client.get("/api/market-impact").json()["car_paths_available"] is True
        assert client.get("/api/market-impact?period=2021").json()["scope"] == "All companies · 2021"
        assert client.get("/api/live/summary").json()["available"] is False
    finally:  # drop the bundle's modules; monkeypatch then restores the project's own
        for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            if Path(getattr(sys.modules[mod], "__file__", "") or "").resolve().is_relative_to(tmp_path.resolve()):
                del sys.modules[mod]


def test_model_card_has_license_and_no_placeholders_after_staging(tmp_path):
    card = (ROOT / "deploy" / "model_card.md").read_text(encoding="utf-8")
    assert "license: cc-by-nc-sa-3.0" in card  # inherited from Financial PhraseBank
    assert "not human-verified" in card
    if all((publish.MODEL_DIR / f).exists() for f in publish.MODEL_FILES):
        publish.stage_model(tmp_path, "someone")
        staged = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert "<hf-user>" not in staged and "someone/distilbert-financial-headline-sentiment" in staged
