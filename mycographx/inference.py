"""Model loading, validation and reproducible uncertainty-aware inference."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import Crippen, Descriptors, Lipinski
from torch_geometric.data import Batch

from .model import EDGE_DIM, NODE_DIM, EEGNNet, smiles_to_graph


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "mico_EEGNN.pth"
PARAMS_PATH = ARTIFACTS / "mico_EEGNN.json"
THRESHOLDS_PATH = ARTIFACTS / "thresholds.json"

TASKS = [
    ("M. fortuitum", "M-fortuitum"),
    ("M. kansasii", "M-kansasii"),
    ("M. bovis", "M-bovis"),
    ("M. tuberculosis (MABA)", "M-tuberculosis-MABA"),
    ("M. tuberculosis (LORA)", "M-tuberculosis-LORA"),
    ("M. smegmatis", "M-smegmatis"),
    ("M. abscessus", "M-abcessus"),
    ("M. tuberculosis (BACTEC)", "M-tuberculosis-BACTEC"),
    ("M. tuberculosis (dilution)", "M-tuberculosis-Dillution"),
]

DEFAULT_EPISTEMIC_PASSES = 30
EMBEDDING_DOMAIN_SCHEMA = "embedding-kde-v4"
_DROPOUT_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)


@dataclass
class Prediction:
    compound_id: str
    input_smiles: str
    canonical_smiles: str
    probabilities: np.ndarray
    epistemic_uncertainties: np.ndarray
    mc_passes: int
    descriptors: dict
    molecule: object
    graph: object
    embedding: np.ndarray
    applicability: object = None

    @property
    def standard_deviations(self):
        """Backward-compatible alias for older callers."""
        return self.epistemic_uncertainties


def _load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = dict(state)
    for prefix in ("module.", "_orig_mod."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix):]: value for key, value in state.items()}
    if not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise TypeError("The .pth file does not contain a valid state_dict.")
    return state


def load_predictor(device_name="cpu"):
    for path in (MODEL_PATH, PARAMS_PATH, THRESHOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path.name}")

    device = torch.device(device_name)
    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as handle:
        thresholds_payload = json.load(handle)

    state = _load_state(MODEL_PATH)
    output_weight = state.get("output_layer.weight")
    if output_weight is None or output_weight.ndim != 2:
        raise ValueError("The checkpoint is incompatible with the EEGNN architecture.")
    num_tasks = int(output_weight.shape[0])
    if num_tasks != len(TASKS):
        raise ValueError(
            f"The checkpoint has {num_tasks} outputs, but the application "
            f"defines {len(TASKS)}."
        )

    agg_dims = [
        int(params[f"agg_hidden_dim_{index + 1}"])
        for index in range(int(params["num_agg_layers"]))
    ]
    linear_dims = [
        int(params[f"lin_hidden_dim_{index + 1}"])
        for index in range(int(params["num_lin_layers"]))
    ]
    model = EEGNNet(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        agg_hidden_dims=agg_dims,
        num_agg_layers=int(params["num_agg_layers"]),
        lin_hidden_dims=linear_dims,
        num_lin_layers=int(params["num_lin_layers"]),
        activation=str(params["activation"]),
        dropout_rate=float(params["dropout_rate"]),
        num_components=int(params["num_components"]),
        concentration=float(params["concentration"]),
        prior_concentration=float(params["prior_concentration"]),
        mcmc_iters=int(params["mcmc_iters"]),
        num_tasks=num_tasks,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    thresholds = np.asarray(thresholds_payload["test"], dtype=float)
    if thresholds.shape != (num_tasks,) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("The threshold JSON does not contain nine valid values.")
    return model, thresholds, device


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_seed(canonical_smiles):
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31 - 1)


def _reset_generators(model):
    for layer in model.agg_layers:
        layer.generator = None


def _activate_mc_dropout(model):
    """Keep evaluation behavior while enabling only dropout sampling."""
    previous_model_mode = model.training
    model.eval()
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            dropout_layers.append((module, module.training))
            module.train(True)
    if not dropout_layers:
        model.train(previous_model_mode)
        raise RuntimeError(
            "The model has no dropout layers for epistemic uncertainty estimation."
        )
    return previous_model_mode, dropout_layers


def _restore_model_mode(model, previous_model_mode, dropout_layers):
    for module, previous_layer_mode in dropout_layers:
        module.train(previous_layer_mode)
    model.train(previous_model_mode)


def _descriptors(mol):
    return {
        "Molecular weight": float(Descriptors.MolWt(mol)),
        "cLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Formal charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "Atoms": int(mol.GetNumAtoms()),
    }


def deterministic_embedding(model, device, canonical_smiles, graph):
    """Extract a reproducible evaluation-mode embedding for AD calculations."""
    batch = Batch.from_data_list([graph]).to(device)
    previous_mode = model.training
    model.eval()
    _seed_everything(_stable_seed(canonical_smiles) + 1_000_003)
    _reset_generators(model)
    try:
        with torch.no_grad():
            embedding = model(batch, return_embedding=True)
    finally:
        model.train(previous_mode)
    if embedding.shape != (1, int(model.embedding_dim)):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    return embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)


def _prediction_uncertainties(prediction):
    values = getattr(prediction, "epistemic_uncertainties", None)
    if values is None:
        values = getattr(prediction, "standard_deviations", None)
    if values is None:
        raise ValueError(
            "This prediction has no epistemic uncertainty values. Run the "
            "prediction again with at least two Monte Carlo passes."
        )
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (len(TASKS),):
        raise ValueError(
            f"Expected {len(TASKS)} epistemic uncertainty values; "
            f"received shape {values.shape}."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Epistemic uncertainty values must be finite and non-negative.")
    return values


def threshold_adjusted_score(probability, threshold):
    """Map a task-specific decision threshold to a common score of 0.5.

    The mapping is piecewise linear and preserves the endpoints: probabilities
    from zero to the threshold occupy [0, 0.5], while probabilities from the
    threshold to one occupy [0.5, 1]. This is a display score, not a calibrated
    probability.
    """
    probability = float(probability)
    threshold = float(threshold)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be a finite value between zero and one.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be a finite value between zero and one.")

    if probability <= threshold:
        return 0.5 if threshold == 0.0 else 0.5 * probability / threshold
    return 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)


def predict_one(
    model,
    device,
    smiles,
    compound_id="Compound",
    stochastic_passes=DEFAULT_EPISTEMIC_PASSES,
):
    stochastic_passes = int(stochastic_passes)
    if stochastic_passes < 2:
        raise ValueError(
            "At least two Monte Carlo passes are required to estimate "
            "epistemic uncertainty."
        )
    canonical, molecule, graph = smiles_to_graph(smiles)
    batch = Batch.from_data_list([graph]).to(device)
    samples = []
    base_seed = _stable_seed(canonical)

    previous_model_mode, dropout_layers = _activate_mc_dropout(model)
    try:
        # The EEGNN sampler temporarily enables gradients internally, so use
        # no_grad instead of inference_mode around the public forward passes.
        with torch.no_grad():
            for pass_index in range(stochastic_passes):
                _seed_everything(base_seed + pass_index)
                _reset_generators(model)
                logits = model(batch)
                if logits.shape != (1, len(TASKS)):
                    raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
                samples.append(torch.sigmoid(logits)[0].detach().cpu().numpy())
    finally:
        _restore_model_mode(model, previous_model_mode, dropout_layers)

    values = np.asarray(samples, dtype=float)
    embedding = deterministic_embedding(model, device, canonical, graph)
    return Prediction(
        compound_id=str(compound_id),
        input_smiles=str(smiles),
        canonical_smiles=canonical,
        probabilities=values.mean(axis=0),
        epistemic_uncertainties=values.std(axis=0, ddof=0),
        mc_passes=stochastic_passes,
        descriptors=_descriptors(molecule),
        molecule=molecule,
        graph=graph,
        embedding=embedding,
    )


def prediction_frame(prediction, thresholds):
    epistemic_uncertainties = _prediction_uncertainties(prediction)
    rows = []
    for index, ((display_name, training_name), threshold) in enumerate(
        zip(TASKS, thresholds)
    ):
        probability = float(prediction.probabilities[index])
        row = {
                "Target": display_name,
                "Probability": probability,
                "Threshold-adjusted score": threshold_adjusted_score(
                    probability, threshold
                ),
                "Threshold": float(threshold),
                "Prediction": "Active" if probability >= threshold else "Inactive",
                "Epistemic uncertainty": float(epistemic_uncertainties[index]),
                "Training name": training_name,
            }
        applicability = getattr(prediction, "applicability", None)
        if getattr(applicability, "schema", None) == EMBEDDING_DOMAIN_SCHEMA:
            row["Embedding KDE applicability domain"] = str(
                applicability.kde_statuses[index]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_prediction(prediction, thresholds):
    epistemic_uncertainties = _prediction_uncertainties(prediction)
    row = {
        "ID": prediction.compound_id,
        "original_SMILES": prediction.input_smiles,
        "canonical_SMILES": prediction.canonical_smiles,
    }
    for index, ((display_name, _), threshold) in enumerate(zip(TASKS, thresholds)):
        key = (
            display_name.replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        probability = float(prediction.probabilities[index])
        row[f"probability__{key}"] = probability
        row[f"threshold_adjusted_score__{key}"] = threshold_adjusted_score(
            probability, threshold
        )
        row[f"epistemic_uncertainty__{key}"] = float(
            epistemic_uncertainties[index]
        )
        row[f"threshold__{key}"] = float(threshold)
        row[f"prediction__{key}"] = (
            "Active" if probability >= threshold else "Inactive"
        )
        applicability = getattr(prediction, "applicability", None)
        if getattr(applicability, "schema", None) == EMBEDDING_DOMAIN_SCHEMA:
            row[f"embedding_kde_domain_status__{key}"] = str(
                applicability.kde_statuses[index]
            )
            row[f"embedding_kde_distance__{key}"] = float(
                applicability.embedding_distances[index]
            )
            row[f"embedding_kde_cdf_percentile__{key}"] = float(
                applicability.kde_percentiles[index]
            )
            row[f"embedding_kde_density__{key}"] = float(
                applicability.kde_densities[index]
            )
    return row
