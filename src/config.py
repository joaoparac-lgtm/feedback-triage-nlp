"""Configuração central do projeto, lida a partir de variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv é opcional em produção (variáveis podem vir do ambiente)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Evita oversubscription de CPU / deadlocks de OpenMP e tokenizers no Windows.
# Precisa ser definido ANTES de o torch ser importado.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "2"))

ZERO_SHOT = "zero-shot"
FINETUNED = "finetuned"

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEFAULT_LABELS: tuple[str, ...] = (
    "Cobrança e pagamento",
    "Problema técnico",
    "Entrega e logística",
    "Cancelamento ou reembolso",
    "Elogio",
    "Dúvida sobre produto",
)
DEFAULT_TEMPLATE = "O assunto principal deste texto é: {}."

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_CSV = DATA_DIR / "sample_feedbacks.csv"


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Variável {name} deve ser um inteiro, recebido: {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"Variável {name} deve ser >= {minimum}, recebido: {value}")
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Variável {name} deve ser numérica, recebido: {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Variável {name} deve estar entre 0 e 1, recebido: {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Parâmetros de execução do pipeline (imutáveis)."""

    model_name: str = DEFAULT_MODEL
    mode: str = ZERO_SHOT
    labels: tuple[str, ...] = DEFAULT_LABELS
    hypothesis_template: str = DEFAULT_TEMPLATE
    batch_size: int = 2  # micro-lote do pipeline (pares texto×categoria no zero-shot)
    max_length: int = 256
    device: str = "auto"
    low_confidence_threshold: float = 0.50
    max_rows: int = 5000
    num_threads: int = 2  # threads intra-op do PyTorch
    chunk_rows: int = 1  # linhas por chamada (granularidade da barra de progresso)

    @classmethod
    def from_env(cls) -> Settings:
        """Constrói as configurações a partir do ambiente (``.env`` incluso)."""
        mode = os.getenv("MODEL_MODE", ZERO_SHOT).strip().lower()
        if mode not in {ZERO_SHOT, FINETUNED}:
            raise ValueError(f"MODEL_MODE inválido: {mode!r} (use '{ZERO_SHOT}' ou '{FINETUNED}')")

        raw_labels = os.getenv("CATEGORIES", "")
        labels = tuple(dict.fromkeys(s.strip() for s in raw_labels.split(";") if s.strip()))

        template = os.getenv("HYPOTHESIS_TEMPLATE", DEFAULT_TEMPLATE)
        if "{}" not in template:
            raise ValueError("HYPOTHESIS_TEMPLATE deve conter o marcador '{}'.")

        device = os.getenv("DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"DEVICE inválido: {device!r}")

        return cls(
            model_name=os.getenv("HF_MODEL_NAME", DEFAULT_MODEL).strip(),
            mode=mode,
            labels=labels or DEFAULT_LABELS,
            hypothesis_template=template,
            batch_size=_get_int("BATCH_SIZE", 2),
            max_length=_get_int("MAX_LENGTH", 256, minimum=8),
            device=device,
            low_confidence_threshold=_get_float("LOW_CONFIDENCE_THRESHOLD", 0.50),
            max_rows=_get_int("MAX_ROWS", 5000),
            num_threads=_get_int("TORCH_NUM_THREADS", 2),
            chunk_rows=_get_int("CHUNK_ROWS", 1),
        )
