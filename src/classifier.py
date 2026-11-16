"""Classificador de textos baseado em modelos da Hugging Face.

Dois modos de operação:

* ``zero-shot``: modelo NLI multilíngue; as categorias são definidas em tempo de
  execução, sem necessidade de treino. É o modo padrão.
* ``finetuned``: qualquer checkpoint de ``AutoModelForSequenceClassification``
  já treinado para as suas classes (ex.: um BERTimbau ajustado por você).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ZERO_SHOT, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Resultado de classificação para um único texto."""

    label: str
    score: float
    probabilities: dict[str, float]  # ordenado do mais para o menos provável


class Classifier(Protocol):
    """Interface mínima exigida pelo processador de dados (facilita testes)."""

    def predict(
        self, texts: Sequence[str], labels: Sequence[str] | None = None
    ) -> list[Prediction]: ...


def resolve_device(preference: str = "auto") -> str:
    """Resolve ``auto`` para o melhor dispositivo disponível."""
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prediction(scores: dict[str, float]) -> Prediction:
    ordered = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    label, score = next(iter(ordered.items()))
    return Prediction(label=label, score=float(score), probabilities=ordered)


class TextClassifier:
    """Envolve um ``transformers.pipeline`` com inferência em lote."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._pipe: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> TextClassifier:
        """Baixa (na 1ª vez) e carrega o modelo. Idempotente."""
        if self._pipe is not None:
            return self

        from transformers import pipeline  # import tardio: acelera CLI/testes

        self._limit_threads()
        device = resolve_device(self.settings.device)
        task = (
            "zero-shot-classification"
            if self.settings.mode == ZERO_SHOT
            else "text-classification"
        )
        logger.info("Carregando %s (%s) em %s", self.settings.model_name, task, device)
        try:
            self._pipe = pipeline(task, model=self.settings.model_name, device=device)
        except OSError as exc:
            raise RuntimeError(
                f"Não foi possível carregar o modelo {self.settings.model_name!r}. "
                "Verifique o nome, a conexão com a internet ou o cache local."
            ) from exc
        return self

    def predict(
        self, texts: Sequence[str], labels: Sequence[str] | None = None
    ) -> list[Prediction]:
        """Classifica uma lista de textos (já limpos e não vazios) em lote.

        Os textos são ordenados por tamanho antes da inferência para minimizar
        *padding* dentro de cada batch; a ordem original é restaurada na saída.

        Args:
            texts: Textos a classificar.
            labels: Categorias candidatas (somente modo ``zero-shot``).

        Returns:
            Uma :class:`Prediction` por texto, na mesma ordem da entrada.
        """
        items = list(texts)
        if not items:
            return []
        if any(not isinstance(t, str) or not t.strip() for t in items):
            raise ValueError("predict() recebe apenas strings não vazias; use clean_text antes.")

        self.load()
        self._limit_threads()  # a cada chamada: o Streamlit executa cada rerun em uma nova thread
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        sorted_texts = [items[i] for i in order]

        if self.settings.mode == ZERO_SHOT:
            raw = self._predict_zero_shot(sorted_texts, labels)
        else:
            raw = self._predict_finetuned(sorted_texts)

        result: list[Prediction | None] = [None] * len(items)
        for original_idx, pred in zip(order, raw, strict=True):
            result[original_idx] = pred
        return [p for p in result if p is not None]

    def predict_one(self, text: str, labels: Sequence[str] | None = None) -> Prediction:
        """Atalho para classificar um único texto."""
        return self.predict([text], labels)[0]

    # ------------------------------------------------------------------ internos
    def _limit_threads(self) -> None:
        """Limita as threads intra-op do PyTorch na thread atual.

        ``torch.set_num_threads`` (OpenMP) é por thread: como o Streamlit roda cada
        rerun em uma thread nova, o limite precisa ser reaplicado antes de cada
        inferência. Sem isso, no Windows/CPU o PyTorch pode usar todos os núcleos
        e competir com o servidor, causando lentidão extrema ou aparente travamento.
        """
        import torch

        torch.set_num_threads(self.settings.num_threads)

    def _predict_zero_shot(
        self, texts: list[str], labels: Sequence[str] | None
    ) -> list[Prediction]:
        candidates = list(dict.fromkeys(labels or self.settings.labels))
        if len(candidates) < 2:
            raise ValueError("Informe ao menos 2 categorias distintas.")
        assert self._pipe is not None
        out = self._pipe(
            texts,
            candidate_labels=candidates,
            hypothesis_template=self.settings.hypothesis_template,
            multi_label=False,  # softmax entre categorias: probabilidades somam 1
            batch_size=self.settings.batch_size,
        )
        if isinstance(out, dict):
            out = [out]
        return [_build_prediction(dict(zip(o["labels"], o["scores"], strict=True))) for o in out]

    def _predict_finetuned(self, texts: list[str]) -> list[Prediction]:
        assert self._pipe is not None
        out = self._pipe(
            texts,
            top_k=None,  # devolve a probabilidade de todas as classes
            batch_size=self.settings.batch_size,
            truncation=True,
            max_length=self.settings.max_length,
        )
        return [_build_prediction({d["label"]: d["score"] for d in per_text}) for per_text in out]
