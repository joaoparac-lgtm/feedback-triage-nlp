"""Limpeza leve de texto, adequada a modelos transformer *cased*."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_SPACE_RE = re.compile(r"\s+")

DEFAULT_MAX_CHARS = 2000


def clean_text(value: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Normaliza um texto bruto de cliente.

    Transformers já lidam bem com pontuação e capitalização, então a limpeza é
    deliberadamente conservadora: normaliza Unicode, remove URLs e caracteres de
    controle, colapsa espaços e limita o tamanho. Valores nulos viram ``""``.

    Args:
        value: Qualquer valor vindo do CSV (``str``, ``NaN``, ``None``, número...).
        max_chars: Limite de caracteres, para evitar entradas gigantes.

    Returns:
        Texto limpo (possivelmente vazio).
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _URL_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:max_chars].strip()
