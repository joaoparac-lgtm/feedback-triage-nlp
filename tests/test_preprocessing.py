import math

import pandas as pd

from src.preprocessing import clean_text


def test_handles_null_values() -> None:
    assert clean_text(None) == ""
    assert clean_text(math.nan) == ""
    assert clean_text(pd.NA) == ""


def test_removes_urls_and_collapses_spaces() -> None:
    assert clean_text("Veja   https://x.com/a?b=1   isso\n\n aqui") == "Veja isso aqui"


def test_truncates_long_text() -> None:
    assert len(clean_text("a" * 5000, max_chars=100)) == 100


def test_keeps_accents_and_case() -> None:
    assert clean_text("  Não funciona!  ") == "Não funciona!"
