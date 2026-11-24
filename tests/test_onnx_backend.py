"""ONNX export / quantization / inference with a tiny randomly initialised DistilBERT (no download)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from src.inference import load_fine_tuned_model, predict_sentiment_batch  # noqa: E402
from src.onnx_backend import (  # noqa: E402
    OnnxSentimentModel,
    export_onnx,
    is_onnx_dir,
    predict_sentiment_batch_onnx,
    quantize_onnx,
    softmax,
)

HEADLINES = ["Tesla deliveries beat estimates", "Intel cuts guidance", "Apple", "",
             "a much longer headline about quarterly earnings guidance and revenue growth that keeps going on"]


def _tiny_model_dir(path, id2label=None):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    words = sorted({w for h in HEADLINES for w in h.lower().split()})
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words)}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(str(path))

    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=2, n_heads=2, hidden_dim=64,
                           max_position_embeddings=128, num_labels=3,
                           id2label=id2label or {0: "negative", 1: "neutral", 2: "positive"},
                           label2id={v: k for k, v in (id2label or {0: "negative", 1: "neutral",
                                                                     2: "positive"}).items()})
    DistilBertForSequenceClassification(cfg).eval().save_pretrained(str(path))
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx")
    src = _tiny_model_dir(root / "torch")
    export_onnx(src, root / "fp32")
    quantize_onnx(root / "fp32", root / "int8")
    return root


def test_export_writes_model_and_side_files(exported):
    for name in ("fp32", "int8"):
        assert is_onnx_dir(exported / name)
        assert (exported / name / "tokenizer.json").exists()
        assert (exported / name / "config.json").exists()
    assert not is_onnx_dir(exported / "torch")


def test_onnx_fp32_matches_pytorch(exported):
    texts = [h for h in HEADLINES if h]
    model, tokenizer, device = load_fine_tuned_model(exported / "torch", device=torch.device("cpu"))
    t_labels, t_probs, t_conf, t_scores = predict_sentiment_batch(texts, model, tokenizer, device, max_length=16)
    o_labels, o_probs, o_conf, o_scores = predict_sentiment_batch_onnx(texts, OnnxSentimentModel(exported / "fp32"),
                                                                       batch_size=2, max_length=16)
    assert o_labels == t_labels
    np.testing.assert_allclose(o_probs, t_probs, atol=1e-5)
    np.testing.assert_allclose(o_scores, o_probs[:, 2] - o_probs[:, 0], atol=1e-6)
    np.testing.assert_allclose(o_conf, o_probs.max(axis=1))


@pytest.mark.parametrize("variant", ["fp32", "int8"])
def test_padding_does_not_change_scores(exported, variant):
    """A short headline must score the same alone and when padded next to a long one."""
    model = OnnxSentimentModel(exported / variant)
    alone = model.logits(["Apple"], max_length=32)
    padded = model.logits(["Apple", HEADLINES[-1]], batch_size=2, max_length=32)[:1]
    np.testing.assert_allclose(alone, padded, atol=1e-4)


def test_int8_returns_valid_probabilities(exported):
    labels, probs, _, scores = predict_sentiment_batch_onnx(HEADLINES[:3], OnnxSentimentModel(exported / "int8"))
    assert len(labels) == 3 and set(labels) <= {"negative", "neutral", "positive"}
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all((scores >= -1) & (scores <= 1))


def test_truncation_and_empty_input(exported):
    model = OnnxSentimentModel(exported / "fp32")
    assert model.logits([HEADLINES[-1]], max_length=8).shape == (1, 3)
    labels, probs, _, _ = predict_sentiment_batch_onnx([], model)
    assert labels == [] and probs.shape == (0, 3)


def test_third_party_label_order(tmp_path):
    src = _tiny_model_dir(tmp_path / "torch", id2label={0: "positive", 1: "negative", 2: "neutral"})
    export_onnx(src, tmp_path / "onnx")
    model = OnnxSentimentModel(tmp_path / "onnx")
    raw = softmax(model.logits(["Apple"]).astype(np.float64))[0]
    _, probs, _, _ = predict_sentiment_batch_onnx(["Apple"], model)
    np.testing.assert_allclose(probs[0], raw[[1, 2, 0]], atol=1e-6)


def test_service_picks_onnx_backend(exported, monkeypatch):
    from src.service import SentimentService

    monkeypatch.setenv("SENTIMENT_BACKEND", "auto")
    service = SentimentService(model_source=str(exported / "int8"))
    assert service.backend == "onnx"
    preds = service.predict(["Tesla deliveries beat estimates"], ticker="TSLA")
    assert preds[0].relevant is True
    assert abs(sum(preds[0].probabilities.values()) - 1) < 1e-3
    assert SentimentService(model_source=str(exported / "torch")).backend == "torch"


def test_self_check_passes_and_catches_a_broken_model(exported):
    import json

    from src.onnx_backend import REFERENCE_FILE, OnnxSelfCheckError, write_reference

    write_reference(exported / "fp32", HEADLINES[:4], [exported / "fp32"], max_length=16)
    model = OnnxSentimentModel(exported / "fp32")
    assert model.self_check() == 1.0
    assert OnnxSentimentModel(exported / "int8").self_check() is None  # no reference file: skipped

    ref_path = exported / "fp32" / REFERENCE_FILE
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    flip = {"negative": "positive", "neutral": "negative", "positive": "neutral"}
    ref["labels"] = [flip[x] for x in ref["labels"]]  # what a broken int8 kernel looks like: nothing agrees
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    with pytest.raises(OnnxSelfCheckError, match="agrees with the fp32 reference on only 0%"):
        model.self_check()
    ref_path.unlink()


def test_invalid_backend_rejected(monkeypatch):
    from src.service import requested_backend

    monkeypatch.setenv("SENTIMENT_BACKEND", "tensorflow")
    with pytest.raises(ValueError):
        requested_backend()
