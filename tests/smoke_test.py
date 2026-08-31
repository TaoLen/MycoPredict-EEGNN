from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mycographx.inference import (
    ARTIFACTS,
    MODEL_PATH,
    flatten_prediction,
    load_predictor,
    predict_one,
    prediction_frame,
    threshold_adjusted_score,
)
from mycographx.applicability import load_applicability_domain
from mycographx.explainability import explain_prediction
from mycographx.model import EDGE_DIM, NODE_DIM, smiles_to_graph


def main():
    canonical, _, graph = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    assert canonical == "CC(=O)Oc1ccccc1C(=O)O"
    assert tuple(graph.x.shape)[1] == NODE_DIM == 48
    assert tuple(graph.edge_attr.shape)[1] == EDGE_DIM == 12

    model, thresholds, device = load_predictor("cpu")
    dropout_modes = []
    handles = [
        module.register_forward_pre_hook(
            lambda current_module, _: dropout_modes.append(current_module.training)
        )
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    try:
        first = predict_one(model, device, canonical, stochastic_passes=4)
    finally:
        for handle in handles:
            handle.remove()
    assert dropout_modes and any(dropout_modes)
    assert not all(dropout_modes)
    assert dropout_modes[-1] is False
    second = predict_one(model, device, canonical, stochastic_passes=4)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(
        first.epistemic_uncertainties, second.epistemic_uncertainties
    )
    np.testing.assert_array_equal(
        first.standard_deviations, first.epistemic_uncertainties
    )
    np.testing.assert_array_equal(first.embedding, second.embedding)
    assert first.probabilities.shape == (9,)
    assert first.epistemic_uncertainties.shape == (9,)
    assert first.mc_passes == 4
    assert first.embedding.shape == (364,)
    assert np.isfinite(first.embedding).all()
    assert np.all((first.probabilities >= 0) & (first.probabilities <= 1))
    assert np.all(first.epistemic_uncertainties >= 0)
    assert np.any(first.epistemic_uncertainties > 0)
    assert model.training is False
    assert not any(
        module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)
    )
    assert thresholds.shape == (9,)
    result = prediction_frame(first, thresholds)
    assert list(result.columns) == [
        "Target",
        "Probability",
        "Threshold-adjusted score",
        "Threshold",
        "Prediction",
        "Epistemic uncertainty",
        "Training name",
    ]
    assert set(result["Prediction"]) == {"Inactive"}
    np.testing.assert_allclose(
        result["Threshold-adjusted score"].to_numpy(),
        [
            threshold_adjusted_score(probability, threshold)
            for probability, threshold in zip(first.probabilities, thresholds)
        ],
    )
    assert np.array_equal(
        result["Threshold-adjusted score"].to_numpy() >= 0.5,
        result["Prediction"].to_numpy() == "Active",
    )
    assert threshold_adjusted_score(0.0, 0.13) == 0.0
    assert threshold_adjusted_score(0.13, 0.13) == 0.5
    assert threshold_adjusted_score(1.0, 0.13) == 1.0
    np.testing.assert_allclose(threshold_adjusted_score(0.907, 0.13), 0.9465517241)
    exported = flatten_prediction(first, thresholds)
    assert "original_SMILES" in exported
    assert "canonical_SMILES" in exported
    assert sum(
        column.startswith("epistemic_uncertainty__") for column in exported
    ) == 9
    assert sum(
        column.startswith("threshold_adjusted_score__") for column in exported
    ) == 9
    assert not any(column.startswith("mc_std__") for column in exported)
    assert not any("probabilidade" in column for column in exported)

    domain = load_applicability_domain(
        ARTIFACTS / "embedding_domain.npz",
        ARTIFACTS / "embedding_domain_metadata.json",
        model_path=MODEL_PATH,
    )
    first.applicability = domain.assess(first)
    assert first.applicability.schema == "embedding-kde-v4"
    assert first.applicability.kde_statuses.shape == (9,)
    assert set(first.applicability.kde_statuses) <= {
        "Inside",
        "Borderline",
        "Outside",
    }
    assert np.all(
        (first.applicability.kde_percentiles >= 0)
        & (first.applicability.kde_percentiles <= 100)
    )
    assert np.all(np.isfinite(first.applicability.kde_densities))
    assert np.all(first.applicability.kde_densities >= 0)
    assert first.applicability.labeled_reference_counts.tolist() == [
        324,
        513,
        622,
        8577,
        1210,
        1772,
        265,
        589,
        3378,
    ]
    task_frame = domain.task_frame(first.applicability)
    assert list(task_frame.columns) == [
        "Target",
        "KDE domain status",
        "Mean 5-NN embedding distance",
        "KDE CDF percentile",
        "Gaussian KDE density",
        "Labeled embedding references",
    ]
    assert not any(
        token in " ".join(task_frame.columns).casefold()
        for token in ("structural", "morgan", "tanimoto", "fingerprint")
    )
    domain_result = prediction_frame(first, thresholds)
    assert "Embedding KDE applicability domain" in domain_result.columns
    domain_export = flatten_prediction(first, thresholds)
    assert sum(
        column.startswith("embedding_kde_domain_status__")
        for column in domain_export
    ) == 9
    assert sum(
        column.startswith("embedding_kde_distance__")
        for column in domain_export
    ) == 9
    assert sum(
        column.startswith("embedding_kde_cdf_percentile__")
        for column in domain_export
    ) == 9
    assert sum(
        column.startswith("embedding_kde_density__")
        for column in domain_export
    ) == 9
    assert not any(
        column.startswith("applicability_domain__")
        or "structural" in column.casefold()
        for column in domain_export
    )
    current_assessment = first.applicability
    first.applicability = type(
        "LegacyHybridAssessment",
        (),
        {
            "statuses": np.asarray(["Outside"] * 9),
            "embedding_zones": np.asarray(["Inside"] * 9),
            "structural_zones": np.asarray(["Outside"] * 9),
        },
    )()
    legacy_result = prediction_frame(first, thresholds)
    assert "Embedding KDE applicability domain" not in legacy_result.columns
    legacy_export = flatten_prediction(first, thresholds)
    assert not any("applicability" in column for column in legacy_export)
    first.applicability = current_assessment
    neighbors = domain.neighbor_frame(first.applicability, task_index=8)
    assert len(neighbors) == 5
    assert neighbors["Embedding distance"].is_monotonic_increasing
    for task_index, kde in enumerate(domain.task_kdes):
        assert kde.bandwidth > 0
        assert 0 < kde.inside_boundary < kde.outer_boundary
        np.testing.assert_allclose(kde.cdf(kde.inside_boundary), 0.95, atol=1e-6)
        np.testing.assert_allclose(kde.cdf(kde.outer_boundary), 0.99, atol=1e-6)
        evaluation_grid = np.linspace(0, kde.outer_boundary * 1.2, 100)
        cdf_values = kde.cdf(evaluation_grid)
        assert np.all(np.diff(cdf_values) >= -1e-12)

        distribution = domain.kde_distribution(first.applicability, task_index)
        assert distribution.histogram["Reference count"].sum() == (
            first.applicability.labeled_reference_counts[task_index]
        )
        assert len(distribution.curve) == 320
        assert len(distribution.boundaries) == 2
        assert len(distribution.query) == 1
        assert np.isfinite(distribution.curve.to_numpy()).all()
        assert np.all(distribution.curve["Gaussian KDE density"] >= 0)
        chart = (
            alt.Chart(distribution.histogram)
            .mark_bar()
            .encode(
                x="Distance lower:Q",
                x2="Distance upper:Q",
                y="Histogram density:Q",
            )
            + alt.Chart(distribution.curve)
            .mark_line()
            .encode(x="Embedding distance:Q", y="Gaussian KDE density:Q")
            + alt.Chart(distribution.query)
            .mark_rule()
            .encode(x="Embedding distance:Q")
        )
        chart.to_dict()

    try:
        predict_one(model, device, canonical, stochastic_passes=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Epistemic uncertainty requires at least two passes.")

    explanation = explain_prediction(
        model=model,
        device=device,
        graph=first.graph,
        molecule=first.molecule,
        canonical_smiles=first.canonical_smiles,
        task_index=8,
    )
    assert explanation.normalized_contributions.shape == (13,)
    assert explanation.relative_contributions.shape == (13,)
    assert explanation.generated_transformations > 0
    assert explanation.evaluated_transformations > 0
    assert (
        explanation.evaluated_transformations
        + explanation.failed_transformations
        + explanation.unchanged_transformations
        == explanation.generated_transformations
    )
    assert explanation.png.startswith(b"\x89PNG")
    assert b"<svg" in explanation.svg[:1000]
    assert explanation.relative_png.startswith(b"\x89PNG")
    assert b"<svg" in explanation.relative_svg[:1000]
    assert explanation.base_probability < thresholds[8]
    assert np.all(explanation.raw_contributions > 0)
    assert np.any(explanation.relative_contributions > 0)
    assert np.any(explanation.relative_contributions < 0)
    assert np.all(explanation.transformation_table["Delta"].to_numpy() > 0)
    assert list(explanation.atom_table.columns) == [
        "Atom",
        "Element",
        "Score effect",
        "Normalized effect",
        "Relative contrast",
        "Std dev",
        "Transformations",
    ]
    assert list(explanation.transformation_table.columns) == [
        "Rule",
        "Family",
        "Focus atom",
        "Affected atoms",
        "Counterfactual score",
        "Delta",
    ]

    try:
        smiles_to_graph("not-a-smiles")
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid SMILES should be rejected.")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
