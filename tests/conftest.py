"""Fixtures compartilhadas: um classificador falso, sem baixar modelo algum."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.classifier import Prediction


class FakeClassifier:
    """Classifica por palavra-chave e registra os textos recebidos."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(
        self, texts: Sequence[str], labels: Sequence[str] | None = None
    ) -> list[Prediction]:
        self.seen.extend(texts)
        out = []
        for t in texts:
            if "cancel" in t.lower():
                probs = {"Cancelamento": 0.9, "Outros": 0.1}
            elif "?" in t:
                probs = {"Dúvida": 0.4, "Outros": 0.6}
            else:
                probs = {"Elogio": 0.8, "Outros": 0.2}
            label = max(probs, key=probs.__getitem__)
            out.append(Prediction(label=label, score=probs[label], probabilities=probs))
        return out


@pytest.fixture
def fake() -> FakeClassifier:
    return FakeClassifier()
