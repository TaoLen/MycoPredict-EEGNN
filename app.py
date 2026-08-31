from __future__ import annotations

import io
import re
import threading

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from mycographx.applicability import (
    EMBEDDING_DOMAIN_SCHEMA,
    load_applicability_domain,
)
from mycographx.explainability import explain_prediction
from mycographx.inference import (
    ARTIFACTS,
    DEFAULT_EPISTEMIC_PASSES,
    EMBEDDING_DOMAIN_SCHEMA as INFERENCE_EMBEDDING_DOMAIN_SCHEMA,
    MODEL_PATH,
    TASKS,
    flatten_prediction,
    load_predictor,
    predict_one,
    prediction_frame,
)


MODEL_LOCK = threading.Lock()

if EMBEDDING_DOMAIN_SCHEMA != INFERENCE_EMBEDDING_DOMAIN_SCHEMA:
    raise RuntimeError("Embedding-domain schema constants are inconsistent.")


st.set_page_config(
    page_title="MycoGraphX",
    page_icon="MP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #17211d;
        --muted: #66736c;
        --line: #dfe6e1;
        --paper: #fbfcfb;
        --green: #166044;
        --coral: #b84d3c;
        --gold: #a87113;
      }
      .stApp { background: var(--paper); color: var(--ink); }
      .block-container { max-width: 1180px; padding-top: 2.3rem; }
      h1, h2, h3 { letter-spacing: 0 !important; color: var(--ink); }
      h1 { font-size: 2.3rem !important; font-weight: 720 !important; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      [data-testid="stMetric"] {
        border-top: 3px solid var(--green);
        padding: .8rem .2rem .2rem .2rem;
      }
      [data-testid="stMetricLabel"] { color: var(--muted); }
      [data-testid="stMetricLabel"] p,
      [data-testid="stMetricValue"],
      [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
      }
      [data-testid="stMetricValue"],
      [data-testid="stMetricValue"] > div {
        font-size: clamp(1.15rem, 2vw, 1.85rem) !important;
        line-height: 1.18 !important;
      }
      [data-testid="stDataFrame"] { border: 1px solid var(--line); }
      div.stButton button,
      div.stDownloadButton button { height: auto; min-height: 2.5rem; }
      div.stButton button p,
      div.stDownloadButton button p,
      [data-baseweb="select"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
      }
      [data-baseweb="select"] > div { height: auto; min-height: 2.5rem; }
      .full-text-table-container {
        width: 100%;
        max-height: 36rem;
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: .35rem;
        background: white;
      }
      table.full-text-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
        font-size: .86rem;
      }
      table.full-text-table th,
      table.full-text-table td {
        padding: .55rem .65rem;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
      }
      table.full-text-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: #f2f5f3;
        color: var(--ink);
        white-space: nowrap;
        overflow-wrap: normal;
      }
      .model-strip {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: .88rem;
        padding: .6rem 0 1.2rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.2rem;
      }
      .model-strip strong { color: var(--ink); font-weight: 650; }
      .disclaimer {
        border-left: 3px solid var(--gold);
        padding: .75rem 1rem;
        color: var(--muted);
        background: #fffdf7;
        margin-top: 1rem;
      }
      .map-legend {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: .8rem;
        color: var(--muted);
        font-size: .84rem;
        margin: -.25rem 0 .8rem 0;
      }
      .legend-blue, .legend-red {
        width: 2.8rem;
        height: .55rem;
        display: inline-block;
      }
      .legend-blue { background: #2d62b7; }
      .legend-red { background: #bd3d35; }
      .result-active { color: var(--green); font-weight: 700; }
      div.stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
      }
      div.stDownloadButton > button { border-color: #9aa69f; }
      @media (max-width: 640px) {
        .block-container { padding-top: 1.25rem; }
        h1 { font-size: 1.85rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_runtime():
    return load_predictor("cpu")


@st.cache_resource(show_spinner=False)
def get_applicability_runtime(schema):
    if schema != EMBEDDING_DOMAIN_SCHEMA:
        raise ValueError("Unsupported embedding-domain schema.")
    return load_applicability_domain(
        ARTIFACTS / "embedding_domain.npz",
        ARTIFACTS / "embedding_domain_metadata.json",
        model_path=MODEL_PATH,
    )


def load_applicability_runtime_for_ui():
    try:
        return get_applicability_runtime(EMBEDDING_DOMAIN_SCHEMA)
    except Exception as error:
        st.warning(
            "Embedding-domain artifacts could not be loaded. Predictions remain "
            "available, but domain analysis is temporarily disabled."
        )
        st.exception(error)
        return None


def normalized_column(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def find_smiles_column(columns):
    aliases = {
        "smiles",
        "canonicalsmiles",
        "isomericsmiles",
        "molecularsmiles",
        "structure",
    }
    for column in columns:
        if normalized_column(column) in aliases:
            return column
    return None


def find_id_column(columns):
    aliases = {"id", "compoundid", "moleculeid", "name", "catalogid"}
    for column in columns:
        if normalized_column(column) in aliases:
            return column
    return None


def render_full_text_table(frame):
    """Render a scrollable table without truncating long cell contents."""
    table_html = frame.to_html(
        index=False,
        escape=True,
        border=0,
        classes="full-text-table",
    )
    st.markdown(
        f'<div class="full-text-table-container">{table_html}</div>',
        unsafe_allow_html=True,
    )


def render_prediction(prediction, thresholds, show_download=True):
    result = prediction_frame(prediction, thresholds)
    active_count = int((result["Prediction"] == "Active").sum())
    best_index = int(prediction.probabilities.argmax())
    best_target = TASKS[best_index][0]
    best_probability = float(prediction.probabilities[best_index])
    best_uncertainty = float(result.iloc[best_index]["Epistemic uncertainty"])

    st.divider()
    metrics = st.columns(4)
    metrics[0].metric("Highest probability", f"{best_probability:.1%}")
    metrics[1].metric("Top target", best_target)
    metrics[2].metric("Epistemic uncertainty", f"{best_uncertainty:.3f}")
    metrics[3].metric("Active predictions", f"{active_count} of {len(TASKS)}")
    mc_passes = getattr(prediction, "mc_passes", None)
    if mc_passes is not None:
        st.caption(
            f"Epistemic uncertainty estimated from {int(mc_passes)} Monte Carlo "
            "dropout passes."
        )

    molecule_column, table_column = st.columns([0.34, 0.66], gap="large")
    with molecule_column:
        st.subheader("Structure")
        image = Draw.MolToImage(prediction.molecule, size=(520, 360))
        st.image(image, use_container_width=True)
        st.code(prediction.canonical_smiles, language=None, wrap_lines=True)
        descriptor_columns = st.columns(2)
        descriptor_items = list(prediction.descriptors.items())
        for index, (label, value) in enumerate(descriptor_items):
            formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
            descriptor_columns[index % 2].metric(label, formatted)

    with table_column:
        st.subheader("Activity profile")
        visible_columns = [
            "Target",
            "Prediction",
            "Threshold-adjusted score",
            "Probability",
            "Threshold",
            "Epistemic uncertainty",
        ]
        visible = result[visible_columns]
        st.dataframe(
            visible,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probability": st.column_config.NumberColumn(
                    "Original probability",
                    format="%.3f",
                    help="Mean model probability across Monte Carlo dropout passes.",
                ),
                "Threshold-adjusted score": st.column_config.ProgressColumn(
                    "Activity score (cutoff 0.500)",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                    help=(
                        "Activity bar rescaled around the task-specific threshold: "
                        "the original interval from 0 to the threshold occupies "
                        "0 to 0.5, and the interval above the threshold to 1 "
                        "occupies 0.5 to 1. This score is not a probability."
                    ),
                ),
                "Threshold": st.column_config.NumberColumn(
                    "Original threshold",
                    format="%.3f",
                    help="Original task-specific threshold before bar rescaling.",
                ),
                "Epistemic uncertainty": st.column_config.NumberColumn(
                    "Epistemic uncertainty",
                    format="%.3f",
                    help=(
                        "Standard deviation of the predicted probability across "
                        "Monte Carlo dropout passes."
                    ),
                ),
                "Target": st.column_config.TextColumn("Target", width="medium"),
                "Prediction": st.column_config.TextColumn(
                    "Prediction", width="small"
                ),
            },
        )
        st.caption(
            "Each activity bar is rescaled independently: the row's original "
            "threshold is exactly 0.500 on the bar. Scores below 0.500 are "
            "Inactive; scores at or above 0.500 are Active. The activity score "
            "is not a probability."
        )
        if show_download:
            csv_data = pd.DataFrame(
                [flatten_prediction(prediction, thresholds)]
            ).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download result CSV",
                data=csv_data,
                file_name="mycographx_result.csv",
                mime="text/csv",
            )


def render_applicability(prediction, domain, key_prefix):
    assessment = getattr(prediction, "applicability", None)
    if getattr(assessment, "schema", None) != EMBEDDING_DOMAIN_SCHEMA:
        assessment = domain.assess(prediction)
        prediction.applicability = assessment

    st.divider()
    st.subheader("Gaussian-KDE embedding applicability domain")
    st.caption(
        "The model compares the query's mean five-neighbor distance in the full "
        "EEGNN embedding space with a Gaussian KDE fitted to leave-one-out "
        "training distances for each endpoint."
    )
    best_index = int(prediction.probabilities.argmax())
    selected_task = st.selectbox(
        "Endpoint for domain details",
        options=range(len(TASKS)),
        index=best_index,
        format_func=lambda index: TASKS[index][0],
        key=f"{key_prefix}_embedding_domain_target",
    )
    status = str(assessment.kde_statuses[selected_task])
    status_message = (
        f"{TASKS[selected_task][0]}: {status} the Gaussian-KDE embedding domain."
    )
    if status == "Inside":
        st.success(status_message)
    elif status == "Outside":
        st.error(status_message)
    else:
        st.warning(status_message)

    metrics = st.columns(6)
    metrics[0].metric("KDE domain status", status)
    metrics[1].metric(
        "Global KDE status", assessment.global_kde_status
    )
    metrics[2].metric(
        "Mean 5-NN embedding distance",
        f"{assessment.embedding_distances[selected_task]:.3f}",
    )
    metrics[3].metric(
        "KDE CDF percentile",
        f"{assessment.kde_percentiles[selected_task]:.1f}%",
    )
    metrics[4].metric(
        "Gaussian KDE density",
        f"{assessment.kde_densities[selected_task]:.3f}",
    )
    metrics[5].metric(
        "Labeled references",
        f"{assessment.labeled_reference_counts[selected_task]:,}",
    )
    st.caption(
        f"Global mean 5-NN distance: {assessment.global_embedding_distance:.3f} | "
        f"Global KDE percentile: {assessment.global_kde_percentile:.1f}% | "
        f"Global KDE density: {assessment.global_kde_density:.3f}. Higher KDE "
        "percentiles indicate a more unusual, distant molecule."
    )

    task_frame = domain.task_frame(assessment)
    st.dataframe(
        task_frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Target": st.column_config.TextColumn("Target", width="large"),
            "Mean 5-NN embedding distance": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "KDE CDF percentile": st.column_config.NumberColumn(format="%.1f%%"),
            "Gaussian KDE density": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.download_button(
        "Download embedding-domain CSV",
        data=task_frame.to_csv(index=False).encode("utf-8"),
        file_name="mycographx_embedding_domain.csv",
        mime="text/csv",
    )

    density_column = st.container()
    neighbor_column = st.container()
    with density_column:
        st.subheader("Embedding-distance distribution")
        st.caption(
            "Bars show the leave-one-out training-distance histogram. The green "
            "curve is its Gaussian KDE; the red rule marks the query."
        )
        distribution = domain.kde_distribution(assessment, selected_task)
        histogram_chart = (
            alt.Chart(distribution.histogram)
            .mark_bar(color="#9aa69f", opacity=0.42)
            .encode(
                x=alt.X(
                    "Distance lower:Q",
                    title="Mean 5-NN distance in the 364-dimensional embedding",
                ),
                x2="Distance upper:Q",
                y=alt.Y("Histogram density:Q", title="Probability density"),
                tooltip=[
                    alt.Tooltip("Distance lower:Q", title="From", format=".4f"),
                    alt.Tooltip("Distance upper:Q", title="To", format=".4f"),
                    alt.Tooltip("Reference count:Q", title="Training molecules"),
                ],
            )
        )
        kde_curve = (
            alt.Chart(distribution.curve)
            .mark_line(color="#166044", size=3)
            .encode(
                x="Embedding distance:Q",
                y="Gaussian KDE density:Q",
                tooltip=[
                    alt.Tooltip("Embedding distance:Q", format=".4f"),
                    alt.Tooltip("Gaussian KDE density:Q", format=".4f"),
                ],
            )
        )
        boundary_rules = (
            alt.Chart(distribution.boundaries)
            .mark_rule(strokeDash=[7, 5], size=2)
            .encode(
                x="Embedding distance:Q",
                color=alt.Color(
                    "Boundary:N",
                    scale=alt.Scale(
                        domain=["95% KDE boundary", "99% KDE boundary"],
                        range=["#a87113", "#7f342b"],
                    ),
                    title="KDE limits",
                ),
                tooltip=[
                    "Boundary:N",
                    alt.Tooltip("Embedding distance:Q", format=".4f"),
                ],
            )
        )
        query_rule = (
            alt.Chart(distribution.query)
            .mark_rule(color="#b84d3c", size=3)
            .encode(
                x="Embedding distance:Q",
                tooltip=[
                    "Marker:N",
                    alt.Tooltip("Embedding distance:Q", format=".4f"),
                    alt.Tooltip("Gaussian KDE density:Q", format=".4f"),
                ],
            )
        )
        query_point = (
            alt.Chart(distribution.query)
            .mark_point(
                shape="diamond", size=180, filled=True, color="#b84d3c"
            )
            .encode(
                x="Embedding distance:Q",
                y="Gaussian KDE density:Q",
            )
        )
        chart = (
            (histogram_chart + kde_curve + boundary_rules + query_rule + query_point)
            .resolve_scale(color="independent")
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Reflection-corrected Gaussian KDE over "
            f"{distribution.reference_count:,} training distances; Scott "
            f"bandwidth = {distribution.bandwidth:.4f}. Inside covers the first "
            f"95% of KDE probability mass, Borderline spans 95–99%, and Outside "
            f"is above 99%."
        )

    with neighbor_column:
        st.subheader("Nearest labeled embeddings")
        neighbors = domain.neighbor_frame(assessment, selected_task)
        molecules = [Chem.MolFromSmiles(smiles) for smiles in neighbors["SMILES"]]
        legends = [
            f"{compound_id} | D={distance:.3f} | {experimental_class}"
            for compound_id, distance, experimental_class in zip(
                neighbors["ID"],
                neighbors["Embedding distance"],
                neighbors["Experimental class"],
            )
        ]
        grid = Draw.MolsToGridImage(
            molecules,
            molsPerRow=2,
            subImgSize=(270, 205),
            legends=legends,
        )
        neighbor_display = neighbors.drop(columns="Reference index").copy()
        neighbor_display["Embedding distance"] = neighbor_display[
            "Embedding distance"
        ].map(
            lambda value: f"{value:.4f}"
        )
        structure_column, neighbor_table_column = st.columns(
            [0.48, 0.52], gap="large", vertical_alignment="top"
        )
        with structure_column:
            st.image(grid, use_container_width=True)
        with neighbor_table_column:
            render_full_text_table(neighbor_display)
    st.info(
        "Being inside the Gaussian-KDE embedding domain does not guarantee biological "
        "activity or correctness. Outside-domain predictions should be treated "
        "as extrapolations even when their probability is high."
    )


def ensure_embedding_applicability(prediction, domain):
    """Replace any cached legacy assessment with Gaussian-KDE results."""
    assessment = getattr(prediction, "applicability", None)
    if getattr(assessment, "schema", None) != EMBEDDING_DOMAIN_SCHEMA:
        prediction.applicability = domain.assess(prediction)
    return prediction


def render_explanation(prediction, model, device, thresholds):
    st.divider()
    st.subheader("Counterfactual explanation map")
    best_index = int(prediction.probabilities.argmax())
    controls = st.columns([0.78, 0.22], vertical_alignment="bottom")
    with controls[0]:
        target_index = st.selectbox(
            "Target to explain",
            options=range(len(TASKS)),
            index=best_index,
            format_func=lambda index: TASKS[index][0],
            key="explanation_target",
        )
    with controls[1]:
        generate_clicked = st.button(
            "Generate map",
            type="primary",
            use_container_width=True,
        )
    st.caption(
        "The map evaluates every valid rulebook perturbation for every atom in "
        "the molecule. Larger structures can take substantially longer."
    )
    map_mode = st.radio(
        "Map scale",
        options=["Relative contrast", "Absolute direction"],
        index=0,
        horizontal=True,
        key="map_scale",
        help=(
            "Relative contrast compares mapped atoms within this molecule. "
            "Absolute direction preserves the signed score effect."
        ),
    )

    explanation_key = (
        prediction.canonical_smiles,
        int(target_index),
    )
    if generate_clicked:
        try:
            with st.spinner("Evaluating chemical counterfactuals..."):
                with MODEL_LOCK:
                    explanation = explain_prediction(
                        model=model,
                        device=device,
                        graph=prediction.graph,
                        molecule=prediction.molecule,
                        canonical_smiles=prediction.canonical_smiles,
                        task_index=target_index,
                    )
            st.session_state["explanation_result"] = explanation
            st.session_state["explanation_key"] = explanation_key
        except Exception as error:
            st.error("The map could not be generated for this structure.")
            st.exception(error)

    explanation = st.session_state.get("explanation_result")
    stored_key = st.session_state.get("explanation_key")
    if explanation is None or stored_key != explanation_key:
        return

    target_threshold = float(thresholds[int(target_index)])
    class_label = (
        "Active"
        if explanation.base_probability >= target_threshold
        else "Inactive"
    )
    transformation_export = explanation.transformation_table.copy()
    relative_mode = map_mode != "Absolute direction"
    map_image = explanation.relative_png if relative_mode else explanation.png
    map_svg = explanation.relative_svg if relative_mode else explanation.svg
    map_scale_name = "relative" if relative_mode else "absolute"

    map_column, detail_column = st.columns([0.62, 0.38], gap="large")
    with map_column:
        st.image(map_image, use_container_width=True)
        if relative_mode:
            st.markdown(
                """
                <div class="map-legend">
                  <span class="legend-blue"></span><span>lower relative effect</span>
                  <span>contrast across mapped atoms</span>
                  <span class="legend-red"></span><span>higher relative effect</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="map-legend">
                  <span class="legend-blue"></span><span>original lowers the score</span>
                  <span>signed effect vs. counterfactuals</span>
                  <span class="legend-red"></span><span>original raises the score</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with detail_column:
        detail_metrics = st.columns(3)
        detail_metrics[0].metric(
            "Model score",
            f"{explanation.base_probability:.1%}",
        )
        detail_metrics[1].metric(
            "Activity threshold",
            f"{target_threshold:.1%}",
        )
        detail_metrics[2].metric(
            "Final class",
            class_label,
        )
        st.caption(
            f"{explanation.evaluated_transformations} valid counterfactuals "
            f"evaluated from {explanation.generated_transformations} rulebook "
            f"candidates across {prediction.molecule.GetNumAtoms()} atoms for "
            f"{TASKS[target_index][0]}."
        )
        if explanation.failed_transformations or explanation.unchanged_transformations:
            st.caption(
                f"Discarded: {explanation.failed_transformations} invalid/failed; "
                f"{explanation.unchanged_transformations} unchanged."
            )
        if explanation.evaluated_transformations == 0:
            st.warning(
                "The rulebook found no applicable transformation for this structure."
            )
        else:
            deltas = explanation.transformation_table["Delta"]
            if bool((deltas > 0).all()):
                st.warning(
                    "All evaluated counterfactuals lowered the score. The "
                    "absolute-direction map is therefore entirely positive. "
                    "In relative-contrast mode, blue marks below-average "
                    "positive effects, not score decreases."
                )
            elif bool((deltas < 0).all()):
                st.warning(
                    "All evaluated counterfactuals raised the score. The "
                    "absolute-direction map is therefore entirely negative. "
                    "In relative-contrast mode, red marks less-negative "
                    "effects, not score increases."
                )

            contribution_notes = []
            positive = explanation.normalized_contributions > 1e-12
            negative = explanation.normalized_contributions < -1e-12
            if positive.any():
                strongest_positive = int(
                    explanation.normalized_contributions.argmax()
                )
                contribution_notes.append(
                    f"Largest relative score increase: atom "
                    f"{strongest_positive}."
                )
            if negative.any():
                strongest_negative = int(
                    explanation.normalized_contributions.argmin()
                )
                contribution_notes.append(
                    f"Largest relative score decrease: atom "
                    f"{strongest_negative}."
                )
            st.write(" ".join(contribution_notes))

        if relative_mode:
            st.info(
                "Relative contrast centers the non-zero atom effects within "
                "this molecule. Blue and red identify lower and higher local "
                "effects, not negative and positive score directions. Use "
                "Absolute direction or the counterfactual transformation list "
                "to inspect the sign."
            )
        else:
            st.info(
                "Absolute direction preserves the signed score effect. Red "
                "means the original region raises the score; blue means it "
                "lowers the score. The Active/Inactive class still depends on "
                "the calibrated threshold."
            )
        st.download_button(
            "Download SVG map",
            data=map_svg,
            file_name=(
                f"mycographx_map_{map_scale_name}_{TASKS[target_index][1]}.svg"
            ),
            mime="image/svg+xml",
            use_container_width=True,
        )
        st.caption(
            "Positive effect: changing the region lowered the score. Negative "
            "effect: changing the region raised the score. Atom indices follow "
            "RDKit conventions and start at 0."
        )

    if not transformation_export.empty:
        with st.expander(
            f"Counterfactual transformations ({len(transformation_export)})"
        ):
            transformation_display = transformation_export.copy()
            transformation_display["Counterfactual score"] = (
                transformation_display["Counterfactual score"].map(
                    lambda value: f"{value:.4f}"
                )
            )
            transformation_display["Delta"] = transformation_display["Delta"].map(
                lambda value: f"{value:+.5f}"
            )
            render_full_text_table(transformation_display)
            st.download_button(
                "Download counterfactual CSV",
                data=transformation_export.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"mycographx_counterfactuals_{TASKS[target_index][1]}.csv"
                ),
                mime="text/csv",
            )


st.title("MycoGraphX")
st.caption(
    "Antimycobacterial activity prediction from molecular structure."
)
st.markdown(
    """
    <div class="model-strip">
      <span><strong>Model</strong> validated EEGNN</span>
      <span><strong>Input</strong> SMILES</span>
      <span><strong>Outputs</strong> 9 predictions + epistemic uncertainty</span>
      <span><strong>Reliability</strong> Gaussian-KDE embedding domain</span>
      <span><strong>Explanation</strong> atom-level counterfactuals</span>
      <span><strong>Runtime</strong> CPU</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, thresholds, device = get_runtime()
except Exception as error:
    st.error(
        "The model could not be loaded. Check the application artifacts."
    )
    st.exception(error)
    st.stop()

applicability_domain = None

with st.sidebar:
    st.header("Settings")
    stochastic_passes = st.slider(
        "Monte Carlo dropout passes",
        min_value=2,
        max_value=50,
        value=DEFAULT_EPISTEMIC_PASSES,
        help=(
            "Controls the epistemic uncertainty estimate. More passes provide "
            "a more stable estimate at the cost of additional processing time."
        ),
    )
    st.caption(f"Device: {device}")

single_tab, batch_tab, model_tab = st.tabs(
    ["Single structure", "CSV file", "Model"]
)

with single_tab:
    smiles = st.text_area(
        "SMILES",
        value="CC(=O)Oc1ccccc1C(=O)O",
        height=105,
        placeholder="Enter a structure in SMILES format",
    )
    predict_clicked = st.button(
        "Run prediction", type="primary", use_container_width=False
    )
    if predict_clicked:
        try:
            with st.spinner("Processing structure..."):
                with MODEL_LOCK:
                    prediction = predict_one(
                        model,
                        device,
                        smiles,
                        stochastic_passes=stochastic_passes,
                    )
                if applicability_domain is None:
                    applicability_domain = load_applicability_runtime_for_ui()
                if applicability_domain is not None:
                    prediction.applicability = applicability_domain.assess(prediction)
            st.session_state["single_prediction"] = prediction
            st.session_state.pop("explanation_result", None)
            st.session_state.pop("explanation_key", None)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error("Inference failed for this structure.")
            st.exception(error)
    prediction = st.session_state.get("single_prediction")
    if prediction is not None:
        if applicability_domain is None:
            applicability_domain = load_applicability_runtime_for_ui()
        if applicability_domain is not None:
            ensure_embedding_applicability(prediction, applicability_domain)
        render_prediction(prediction, thresholds)
        if applicability_domain is not None:
            render_applicability(
                prediction, applicability_domain, key_prefix="single"
            )
        render_explanation(prediction, model, device, thresholds)

with batch_tab:
    uploaded = st.file_uploader(
        "CSV table",
        type=["csv"],
        help="The table must contain a SMILES column and may contain an ID column.",
    )
    example = pd.DataFrame(
        {
            "ID": ["example_1", "example_2"],
            "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
        }
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sample CSV",
        data=example,
        file_name="sample_input.csv",
        mime="text/csv",
    )

    if uploaded is not None:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("The file exceeds the 5 MB limit.")
        else:
            try:
                table = pd.read_csv(
                    io.BytesIO(uploaded.getvalue()), sep=None, engine="python"
                )
                smiles_column = find_smiles_column(table.columns)
                id_column = find_id_column(table.columns)
                if smiles_column is None:
                    st.error("No SMILES column was found.")
                elif len(table) > 200:
                    st.error("Each run is limited to 200 structures.")
                elif table.empty:
                    st.error("The file contains no records.")
                else:
                    st.caption(
                        f"{len(table)} structures | SMILES column: {smiles_column}"
                    )
                    st.dataframe(table.head(8), hide_index=True, use_container_width=True)
                    if st.button("Process file", type="primary"):
                        if applicability_domain is None:
                            applicability_domain = load_applicability_runtime_for_ui()
                        progress = st.progress(0, text="Starting...")
                        output_rows = []
                        invalid_rows = []
                        predictions = []
                        for position, (_, row) in enumerate(table.iterrows(), start=1):
                            compound_id = (
                                row[id_column]
                                if id_column is not None
                                else f"compound_{position}"
                            )
                            input_smiles = row[smiles_column]
                            try:
                                with MODEL_LOCK:
                                    prediction = predict_one(
                                        model,
                                        device,
                                        input_smiles,
                                        compound_id=compound_id,
                                        stochastic_passes=stochastic_passes,
                                    )
                                if applicability_domain is not None:
                                    prediction.applicability = applicability_domain.assess(
                                        prediction
                                    )
                                predictions.append(prediction)
                                output_rows.append(
                                    flatten_prediction(prediction, thresholds)
                                )
                            except ValueError as error:
                                invalid_rows.append(
                                    {
                                        "ID": compound_id,
                                        "SMILES": input_smiles,
                                        "Error": str(error),
                                    }
                                )
                            progress.progress(
                                position / len(table),
                                text=(
                                    f"Processed {position} of {len(table)} "
                                    "structures"
                                ),
                            )
                        progress.empty()

                        st.success(
                            f"{len(output_rows)} valid structures; "
                            f"{len(invalid_rows)} invalid."
                        )
                        if output_rows:
                            output = pd.DataFrame(output_rows)
                            st.dataframe(
                                output,
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.download_button(
                                "Download predictions CSV",
                                data=output.to_csv(index=False).encode("utf-8"),
                                file_name="mycographx_predictions.csv",
                                mime="text/csv",
                            )
                            selected = st.selectbox(
                                "View structure",
                                options=range(len(predictions)),
                                format_func=lambda index: predictions[index].compound_id,
                            )
                            render_prediction(
                                predictions[selected], thresholds, show_download=False
                            )
                            if applicability_domain is not None:
                                render_applicability(
                                    predictions[selected],
                                    applicability_domain,
                                    key_prefix="batch",
                                )
                        if invalid_rows:
                            invalid = pd.DataFrame(invalid_rows)
                            with st.expander("Invalid records"):
                                st.dataframe(
                                    invalid, hide_index=True, use_container_width=True
                                )
                                st.download_button(
                                    "Download invalid records",
                                    data=invalid.to_csv(index=False).encode("utf-8"),
                                    file_name="mycographx_invalid.csv",
                                    mime="text/csv",
                                )
            except Exception as error:
                st.error(f"The CSV could not be read: {error}")

with model_tab:
    st.subheader("Model scope")
    st.write(
        "The EEGNN checkpoint generates probabilities for nine "
        "antimycobacterial endpoints. Classification uses thresholds calibrated "
        "on the validation set and applied to the test set. Epistemic uncertainty "
        "is estimated as the population standard deviation of predicted "
        "probabilities across Monte Carlo dropout passes. The stochastic EEGNN "
        "graph generator is resampled on every pass. Explanation maps measure "
        "score changes caused by counterfactuals defined in the chemical rulebook. "
        "Applicability status uses a Gaussian KDE of task-specific leave-one-out "
        "five-neighbor distances over final EEGNN embeddings from the training set."
    )
    threshold_table = pd.DataFrame(
        {
            "Target": [display_name for display_name, _ in TASKS],
            "Threshold": thresholds,
        }
    )
    st.dataframe(
        threshold_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Threshold": st.column_config.NumberColumn(
                "Threshold", format="%.2f"
            )
        },
    )
    st.markdown(
        """
        <div class="disclaimer">
          This tool is intended for research and experimental prioritization.
          Results do not constitute a diagnosis, clinical recommendation, or
          confirmation of biological activity. Predictions may be less reliable
          for compounds outside the chemical domain of the training data.
        </div>
        """,
        unsafe_allow_html=True,
    )
