import io

import pandas as pd
import pytest

from src.config import DEFAULT_LABELS, Settings
from src.data_processing import (
    COL_CATEGORY,
    COL_CONFIDENCE,
    COL_REVIEW,
    CSVLoadError,
    classify_dataframe,
    generate_sample_data,
    guess_text_column,
    load_csv,
)


def test_load_csv_semicolon_and_latin1() -> None:
    raw = "id;texto\n1;não funciona\n".encode("latin-1")
    df = load_csv(io.BytesIO(raw))
    assert list(df.columns) == ["id", "texto"]
    assert df.loc[0, "texto"] == "não funciona"


@pytest.mark.parametrize("payload", [b"", b"   \n  "])
def test_load_csv_empty(payload: bytes) -> None:
    with pytest.raises(CSVLoadError):
        load_csv(payload)


def test_load_csv_missing_file(tmp_path) -> None:
    with pytest.raises(CSVLoadError):
        load_csv(tmp_path / "nao_existe.csv")


def test_load_csv_header_only() -> None:
    with pytest.raises(CSVLoadError):
        load_csv(b"id,texto\n")


def test_classify_preserves_order_and_flags(fake) -> None:
    df = pd.DataFrame(
        {"texto": ["Quero cancelar", "Ótimo serviço", None, "Funciona?", "Quero cancelar"]}
    )
    out = classify_dataframe(df, fake, chunk_size=2, low_confidence_threshold=0.5)

    assert out[COL_CATEGORY].tolist()[:2] == ["Cancelamento", "Elogio"]
    assert pd.isna(out.loc[2, COL_CATEGORY]) and pd.isna(out.loc[2, COL_CONFIDENCE])
    assert out.loc[3, COL_CATEGORY] == "Outros"
    assert out[COL_REVIEW].tolist() == [False, False, True, False, False]
    assert COL_CATEGORY not in df.columns  # entrada não é mutada


def test_classify_deduplicates(fake) -> None:
    df = pd.DataFrame({"texto": ["Quero cancelar"] * 10})
    classify_dataframe(df, fake)
    assert fake.seen == ["Quero cancelar"]


def test_classify_progress_callback(fake) -> None:
    calls: list[tuple[int, int]] = []
    df = pd.DataFrame({"texto": [f"texto {i}" for i in range(5)]})
    classify_dataframe(df, fake, chunk_size=2, progress_callback=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (5, 5)


def test_classify_invalid_column(fake) -> None:
    with pytest.raises(CSVLoadError, match="não encontrada"):
        classify_dataframe(pd.DataFrame({"a": ["x"]}), fake, text_column="texto")


def test_classify_all_blank(fake) -> None:
    with pytest.raises(CSVLoadError):
        classify_dataframe(pd.DataFrame({"texto": ["", None, "  "]}), fake)


def test_guess_text_column() -> None:
    assert guess_text_column(pd.DataFrame({"id": ["1"], "Feedback": ["abc"]})) == "Feedback"


def test_generate_sample_is_deterministic() -> None:
    a, b = generate_sample_data(50, seed=1), generate_sample_data(50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["categoria_real"]) <= set(DEFAULT_LABELS)


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CATEGORIES", "A; B ;A")
    monkeypatch.setenv("BATCH_SIZE", "8")
    s = Settings.from_env()
    assert s.labels == ("A", "B") and s.batch_size == 8
    monkeypatch.setenv("BATCH_SIZE", "abc")
    with pytest.raises(ValueError):
        Settings.from_env()
