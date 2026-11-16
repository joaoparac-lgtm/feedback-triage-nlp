"""Interface Streamlit: classificação individual e em lote de feedbacks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite `import src`

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.classifier import Prediction, TextClassifier  # noqa: E402
from src.config import SAMPLE_CSV, ZERO_SHOT, Settings  # noqa: E402
from src.data_processing import (  # noqa: E402
    COL_CATEGORY,
    COL_CONFIDENCE,
    COL_REVIEW,
    CSVLoadError,
    classify_dataframe,
    guess_text_column,
    load_csv,
    validate_text_column,
)
from src.preprocessing import clean_text  # noqa: E402

st.set_page_config(page_title="Triagem de Feedbacks", page_icon="🗂️", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo da Hugging Face (só na 1ª vez)…")
def get_classifier() -> TextClassifier:
    """Carrega o modelo uma única vez por processo (compartilhado entre sessões)."""
    return TextClassifier(Settings.from_env()).load()


def probabilities_chart(pred: Prediction):  # noqa: ANN201
    """Gráfico de barras horizontais com a probabilidade de cada categoria."""
    df = pd.DataFrame(
        {"Categoria": list(pred.probabilities), "Probabilidade": list(pred.probabilities.values())}
    ).sort_values("Probabilidade")
    fig = px.bar(
        df,
        x="Probabilidade",
        y="Categoria",
        orientation="h",
        text=df["Probabilidade"].map("{:.1%}".format),
        range_x=[0, 1],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=60 * len(df) + 80, margin=dict(l=0, r=30, t=10, b=0))
    return fig


def render_sidebar(settings: Settings) -> tuple[list[str], float]:
    """Mostra as configurações e devolve (categorias, limiar de baixa confiança)."""
    st.sidebar.header("⚙️ Configurações")
    st.sidebar.caption(f"Modelo: `{settings.model_name}`  \nModo: `{settings.mode}`")

    if settings.mode == ZERO_SHOT:
        text = st.sidebar.text_area(
            "Categorias (uma por linha)",
            "\n".join(settings.labels),
            height=180,
            help="No modo zero-shot você pode mudar as categorias sem retreinar nada.",
        )
        labels = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
        if len(labels) < 2:
            st.sidebar.error("Informe ao menos 2 categorias.")
            st.stop()
    else:
        labels = []
        st.sidebar.info("Modo fine-tuned: as categorias vêm do próprio modelo.")

    threshold = st.sidebar.slider(
        "Limiar de baixa confiança",
        0.0,
        1.0,
        settings.low_confidence_threshold,
        0.05,
        help="Linhas abaixo deste valor são marcadas para revisão manual.",
    )
    return labels, threshold


def tab_single(classifier: TextClassifier, labels: list[str]) -> None:
    """Aba 1: classificação de um único texto."""
    st.subheader("Classificar um texto")
    text = st.text_area(
        "Feedback do cliente",
        height=140,
        placeholder="Ex.: Fui cobrado duas vezes e ninguém responde meus e-mails…",
    )
    if not st.button("Classificar", type="primary", key="single"):
        return

    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Digite um texto para classificar.")
        return
    try:
        with st.spinner("Analisando…"):
            pred = classifier.predict_one(cleaned, labels or None)
    except (RuntimeError, ValueError) as exc:
        st.error(f"Falha na inferência: {exc}")
        return

    col_metrics, col_chart = st.columns([1, 2])
    col_metrics.metric("Categoria prevista", pred.label)
    col_metrics.metric("Confiança", f"{pred.score:.1%}")
    col_chart.plotly_chart(probabilities_chart(pred))


def tab_batch(
    classifier: TextClassifier, labels: list[str], threshold: float, max_rows: int
) -> None:
    """Aba 2: upload de CSV, execução do pipeline e métricas agregadas."""
    st.subheader("Processar um CSV")
    uploaded = st.file_uploader("Envie um arquivo CSV", type=["csv"])
    if SAMPLE_CSV.exists():
        st.caption("Sem arquivo? Use `data/sample_feedbacks.csv` do repositório.")
    if uploaded is None:
        return

    try:
        df = load_csv(uploaded.getvalue())  # getvalue(): não depende da posição do ponteiro
    except CSVLoadError as exc:
        st.error(f"Não foi possível ler o CSV: {exc}")
        return
    if len(df) > max_rows:
        st.error(f"O arquivo tem {len(df)} linhas; o limite é {max_rows} (variável MAX_ROWS).")
        return

    st.write(f"**{len(df)} linhas** · pré-visualização:")
    st.dataframe(df.head(5), hide_index=True)

    columns = list(df.columns)
    column = st.selectbox("Coluna com o texto", columns, index=columns.index(guess_text_column(df)))

    if st.button("Executar pipeline", type="primary", key="batch"):
        st.caption("⏳ Processando — evite mexer nos widgets até terminar (isso reinicia o script).")
        bar = st.progress(0.0, text="Preparando…")
        started = time.perf_counter()

        def on_progress(done: int, total: int) -> None:
            """Atualiza a barra a cada micro-lote, com tempo decorrido e estimativa."""
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done) if done else 0.0
            text = f"{done}/{total} textos únicos · {elapsed:.0f}s decorridos"
            if done:
                text += f" · ~{eta:.0f}s restantes"
            bar.progress(done / max(total, 1), text=text)

        try:
            validate_text_column(df, column)
            result = classify_dataframe(
                df,
                classifier,
                column,
                labels or None,
                chunk_size=classifier.settings.chunk_rows,
                low_confidence_threshold=threshold,
                progress_callback=on_progress,
            )
        except (CSVLoadError, RuntimeError, ValueError) as exc:
            bar.empty()
            st.error(str(exc))
            return
        bar.empty()
        st.success(f"Concluído em {time.perf_counter() - started:.1f}s.")
        st.session_state["batch_result"] = result

    result: pd.DataFrame | None = st.session_state.get("batch_result")
    if result is None:
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", len(result))
    m2.metric("Classificadas", int(result[COL_CATEGORY].notna().sum()))
    m3.metric("Confiança média", f"{result[COL_CONFIDENCE].mean():.1%}")
    m4.metric("Para revisão manual", int(result[COL_REVIEW].sum()))

    left, right = st.columns(2)
    dist = (
        result[COL_CATEGORY].value_counts().rename_axis("Categoria").reset_index(name="Quantidade")
    )
    left.plotly_chart(
        px.bar(dist, x="Categoria", y="Quantidade", title="Distribuição por categoria")
    )
    right.plotly_chart(
        px.histogram(
            result.dropna(subset=[COL_CONFIDENCE]),
            x=COL_CONFIDENCE,
            nbins=20,
            range_x=[0, 1],
            title="Distribuição da confiança",
        )
    )

    options = sorted(result[COL_CATEGORY].dropna().unique())
    chosen = st.multiselect("Filtrar categorias", options, default=options)
    only_review = st.checkbox("Mostrar apenas linhas para revisão manual")
    view = result[result[COL_CATEGORY].isin(chosen) | result[COL_CATEGORY].isna()]
    if only_review:
        view = view[view[COL_REVIEW]]
    st.dataframe(view, hide_index=True)

    st.download_button(
        "⬇️ Baixar resultado (CSV)",
        data=result.to_csv(index=False).encode("utf-8-sig"),  # BOM: abre certo no Excel
        file_name="feedbacks_classificados.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("🗂️ Triagem Automática de Feedbacks")
    st.caption("Classificação de textos em português com modelos open-source da Hugging Face.")

    try:
        settings = Settings.from_env()
        classifier = get_classifier()
    except (ValueError, RuntimeError) as exc:
        st.error(f"Erro de configuração/modelo: {exc}")
        st.stop()

    labels, threshold = render_sidebar(settings)
    tab1, tab2 = st.tabs(["🔍 Classificador individual", "📦 Processamento em lote"])
    with tab1:
        tab_single(classifier, labels)
    with tab2:
        tab_batch(classifier, labels, threshold, settings.max_rows)


main()
