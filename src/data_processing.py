"""Leitura/validação de CSV, classificação de DataFrames e geração de dados de exemplo."""

from __future__ import annotations

import csv
import io
import logging
import random
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import IO, Any

import pandas as pd

from .classifier import Classifier, Prediction
from .config import DEFAULT_LABELS
from .preprocessing import clean_text

logger = logging.getLogger(__name__)

COL_TEXT = "texto"
COL_CATEGORY = "categoria_prevista"
COL_CONFIDENCE = "confianca"
COL_REVIEW = "requer_revisao"

ProgressCallback = Callable[[int, int], None]


class CSVLoadError(ValueError):
    """Erro amigável para arquivos CSV ausentes, vazios, corrompidos ou inválidos."""


# --------------------------------------------------------------------- leitura
def load_csv(source: str | Path | IO[Any] | bytes) -> pd.DataFrame:
    """Lê um CSV de forma tolerante (encoding e separador detectados).

    Tenta ``utf-8-sig`` e depois ``latin-1``; o separador (``,`` ``;`` tab) é
    inferido automaticamente.

    Raises:
        CSVLoadError: arquivo inexistente, vazio, ilegível ou sem linhas.
    """
    try:
        if isinstance(source, bytes):
            raw = source
        elif isinstance(source, (str, Path)):
            raw = Path(source).read_bytes()
        else:
            data = source.read()
            raw = data.encode("utf-8") if isinstance(data, str) else data
    except FileNotFoundError as exc:
        raise CSVLoadError(f"Arquivo não encontrado: {source}") from exc
    except OSError as exc:
        raise CSVLoadError(f"Não foi possível ler o arquivo: {exc}") from exc

    if not raw.strip():
        raise CSVLoadError("O arquivo está vazio.")

    df: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.StringIO(raw.decode(encoding)), sep=None, engine="python", dtype=str)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
            raise CSVLoadError(f"CSV corrompido ou malformado: {exc}") from exc

    if df is None:
        raise CSVLoadError(f"Não foi possível decodificar o arquivo: {last_error}")
    if df.empty:
        raise CSVLoadError("O CSV não contém linhas de dados.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_text_column(df: pd.DataFrame, column: str) -> None:
    """Garante que a coluna de texto existe e tem ao menos um valor não vazio."""
    if column not in df.columns:
        raise CSVLoadError(
            f"Coluna {column!r} não encontrada. Colunas disponíveis: {list(df.columns)}"
        )
    if not df[column].map(clean_text).astype(bool).any():
        raise CSVLoadError(f"A coluna {column!r} não tem nenhum texto válido.")


def guess_text_column(df: pd.DataFrame) -> str:
    """Sugere a coluna de texto: nome conhecido ou a de maior comprimento médio."""
    known = ("texto", "text", "feedback", "mensagem", "comentario", "comentário", "descricao")
    lowered = {c.lower(): c for c in df.columns}
    for name in known:
        if name in lowered:
            return lowered[name]
    return str(df.astype(str).apply(lambda s: s.str.len().mean()).idxmax())


# ----------------------------------------------------------------- classificação
def classify_dataframe(
    df: pd.DataFrame,
    classifier: Classifier,
    text_column: str = COL_TEXT,
    labels: Sequence[str] | None = None,
    chunk_size: int = 1,
    low_confidence_threshold: float = 0.50,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Classifica uma coluna de texto e devolve uma *cópia* com as novas colunas.

    Otimizações: textos duplicados são classificados uma única vez; textos únicos
    são ordenados por tamanho (menos *padding*) e processados em micro-chunks
    (``chunk_size`` linhas por vez, padrão 1), o que mantém a CPU sob controle e
    permite reportar progresso em tempo real via ``progress_callback(feitos, total)``,
    chamado também com ``(0, total)`` antes da primeira inferência.

    Colunas adicionadas: ``categoria_prevista``, ``confianca`` e
    ``requer_revisao`` (texto vazio ou confiança < limiar).

    Raises:
        CSVLoadError: se ``text_column`` não existir ou estiver totalmente vazia.
    """
    validate_text_column(df, text_column)
    if chunk_size < 1:
        raise ValueError("chunk_size deve ser >= 1")

    result = df.copy()
    cleaned = result[text_column].map(clean_text)
    unique_texts = sorted({t for t in cleaned if t}, key=len)
    total = len(unique_texts)
    logger.info("Classificando %d textos únicos (%d linhas).", total, len(result))

    predictions: dict[str, Prediction] = {}
    if progress_callback:
        progress_callback(0, total)
    for start in range(0, total, chunk_size):
        batch = unique_texts[start : start + chunk_size]
        try:
            batch_preds = classifier.predict(batch, labels)
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao classificar o texto {batch[0][:60]!r}...: {exc}"
            ) from exc
        for text, pred in zip(batch, batch_preds, strict=True):
            predictions[text] = pred
        if progress_callback:
            progress_callback(min(start + chunk_size, total), total)

    result[COL_CATEGORY] = cleaned.map(
        lambda t: predictions[t].label if t in predictions else pd.NA
    )
    result[COL_CONFIDENCE] = pd.to_numeric(
        cleaned.map(lambda t: round(predictions[t].score, 4) if t in predictions else float("nan"))
    )
    result[COL_REVIEW] = result[COL_CONFIDENCE].isna() | (
        result[COL_CONFIDENCE] < low_confidence_threshold
    )
    return result


def to_hf_dataset(df: pd.DataFrame) -> Any:
    """Converte o DataFrame em ``datasets.Dataset`` (Arrow) para salvar/publicar."""
    from datasets import Dataset

    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)


# ------------------------------------------------------------------ dados de exemplo
_TEMPLATES: dict[str, list[str]] = {
    "Cobrança e pagamento": [
        "Fui cobrado duas vezes no cartão de crédito pelo mesmo pedido.",
        "A fatura deste mês veio com um valor diferente do contratado.",
        "Meu boleto não foi compensado e já recebi aviso de cobrança.",
        "O pagamento via Pix foi debitado, mas o sistema continua mostrando pendente.",
        "Não reconheço uma taxa que apareceu na minha fatura.",
    ],
    "Problema técnico": [
        "O aplicativo fecha sozinho toda vez que tento abrir a tela de pedidos.",
        "Não consigo fazer login, aparece erro 500 depois de digitar a senha.",
        "O site está muito lento e a página de checkout trava.",
        "Atualizei o sistema e agora o recurso de relatórios parou de funcionar.",
        "Recebo uma mensagem de erro ao tentar enviar arquivos pela plataforma.",
    ],
    "Entrega e logística": [
        "Meu pedido está atrasado há duas semanas e o rastreio não atualiza.",
        "A transportadora entregou no endereço errado.",
        "O produto chegou com a embalagem violada e amassada.",
        "Recebi apenas um dos três itens que comprei.",
        "O prazo de entrega informado no site não foi cumprido.",
    ],
    "Cancelamento ou reembolso": [
        "Quero cancelar minha assinatura imediatamente.",
        "Solicitei o reembolso há 20 dias e até agora nada foi devolvido.",
        "Desisti da compra e gostaria de devolver o produto.",
        "Como faço para encerrar minha conta e receber o estorno?",
        "Preciso cancelar o contrato, pois não vou mais usar o serviço.",
    ],
    "Elogio": [
        "Atendimento excelente, o suporte resolveu meu problema em minutos!",
        "Adorei o produto, superou minhas expectativas. Parabéns à equipe.",
        "Entrega rápida e tudo perfeito, recomendo a todos.",
        "A plataforma é muito intuitiva, estou muito satisfeito.",
        "Muito obrigado pela atenção, foi o melhor atendimento que já tive.",
    ],
    "Dúvida sobre produto": [
        "Esse modelo é compatível com o meu aparelho?",
        "Qual é a diferença entre o plano básico e o plano premium?",
        "O produto acompanha garantia estendida? Por quanto tempo?",
        "Vocês têm esse item disponível em outras cores ou tamanhos?",
        "Onde encontro o manual de instruções para montar o produto?",
    ],
}
_PREFIXES = ["", "", "Olá, ", "Bom dia! ", "Boa tarde, ", "Prezados, ", "Oi, "]
_SUFFIXES = ["", "", " Aguardo retorno.", " Obrigado.", " Preciso de ajuda urgente.", " Att."]
_CHANNELS = ["e-mail", "chat", "telefone", "redes sociais", "app"]


def generate_sample_data(n: int = 200, seed: int = 42, blank_ratio: float = 0.02) -> pd.DataFrame:
    """Gera feedbacks sintéticos em português para demonstração.

    Inclui ``categoria_real`` (rótulo sintético, útil para avaliar o modelo) e
    uma pequena fração de textos vazios para exercitar o tratamento de erros.
    """
    if n < 1:
        raise ValueError("n deve ser >= 1")
    assert set(_TEMPLATES) == set(DEFAULT_LABELS), "templates fora de sincronia com DEFAULT_LABELS"
    rng = random.Random(seed)
    categories = list(_TEMPLATES)
    start = date(2025, 1, 1)
    rows = []
    for i in range(1, n + 1):
        category = rng.choice(categories)
        text = rng.choice(_PREFIXES) + rng.choice(_TEMPLATES[category]) + rng.choice(_SUFFIXES)
        if rng.random() < blank_ratio:
            text = ""
        rows.append(
            {
                "id": i,
                "data": (start + timedelta(days=rng.randint(0, 364))).isoformat(),
                "canal": rng.choice(_CHANNELS),
                COL_TEXT: text,
                "categoria_real": category,
            }
        )
    return pd.DataFrame(rows)
