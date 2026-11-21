"""Inference tests with a tiny fake model, so no checkpoint download is needed."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.inference import predict_sentiment_batch  # noqa: E402


class FakeTokenizer:
    """Encodes each text as its index so the fake model can return fixed logits."""

    def __call__(self, texts, truncation, padding, max_length, return_tensors):
        ids = torch.arange(len(texts)).unsqueeze(1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel:
    def __init__(self, logits):
        self.logits = torch.tensor(logits, dtype=torch.float)

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.logits[input_ids.squeeze(1)])


def test_predict_sentiment_batch_labels_probs_and_score():
    logits = [
        [5.0, 0.0, 0.0],   # negative
        [0.0, 5.0, 0.0],   # neutral
        [0.0, 0.0, 5.0],   # positive
    ]
    labels, probs, confs, scores = predict_sentiment_batch(
        ["a", "b", "c"], FakeModel(logits), FakeTokenizer(), torch.device("cpu"), batch_size=2
    )

    assert labels == ["negative", "neutral", "positive"]
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(confs, probs.max(axis=1))
    np.testing.assert_allclose(scores, probs[:, 2] - probs[:, 0])
    assert scores[0] < 0 < scores[2]
    assert abs(scores[1]) < 1e-6
    assert np.all((scores >= -1) & (scores <= 1))


def test_predict_sentiment_batch_empty_input():
    labels, probs, confs, scores = predict_sentiment_batch(
        [], FakeModel([[0.0, 0.0, 0.0]]), FakeTokenizer(), torch.device("cpu")
    )
    assert labels == []
    assert probs.shape == (0, 3)
    assert len(scores) == 0
