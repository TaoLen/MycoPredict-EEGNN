"""Build a Gaussian-KDE applicability domain from final EEGNN embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mycographx.applicability import (  # noqa: E402
    EMBEDDING_DOMAIN_SCHEMA,
    INSIDE_CDF_PROBABILITY,
    NEIGHBOR_COUNT,
    OUTER_CDF_PROBABILITY,
    GaussianDistanceKDE,
    _zone_from_cdf,
)
from mycographx.inference import (  # noqa: E402
    MODEL_PATH,
    TASKS,
    deterministic_embedding,
    load_predictor,
)
from mycographx.model import smiles_to_graph  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Training CSV containing SMILES and endpoint labels.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="Validation CSV containing SMILES and endpoint labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "mycographx" / "artifacts" / "embedding_domain.npz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "mycographx" / "artifacts" / "embedding_domain_metadata.json",
    )
    return parser.parse_args()


def prepare_structures(table, model, device, label):
    count = len(table)
    canonical_smiles = []
    embeddings = np.zeros((count, int(model.embedding_dim)), dtype=np.float32)
    for position, smiles in enumerate(table["SMILES"].astype(str), start=1):
        canonical, _, graph = smiles_to_graph(smiles)
        canonical_smiles.append(canonical)
        embeddings[position - 1] = deterministic_embedding(
            model, device, canonical, graph
        )
        if position == 1 or position % 250 == 0 or position == count:
            print(f"{label}: {position}/{count}", flush=True)
    if len(set(canonical_smiles)) != count:
        raise ValueError(f"{label} contains duplicate canonical structures.")
    return np.asarray(canonical_smiles, dtype=str), embeddings


def normalize_embeddings(embeddings, mean, scale):
    standardized = (embeddings - mean) / scale
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    return np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized, dtype=np.float32),
        where=norms > 1e-12,
    ).astype(np.float32)


def embedding_loo_distances(embeddings, indices):
    references = embeddings[indices]
    neighbors = NearestNeighbors(
        n_neighbors=NEIGHBOR_COUNT + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    ).fit(references)
    distances, _ = neighbors.kneighbors(references, return_distance=True)
    return distances[:, 1:].mean(axis=1).astype(np.float32)


def query_distance_summary(
    query_embeddings,
    query_labels,
    train_embeddings,
    train_labels,
    task_kdes,
):
    summaries = []
    for task_index, (display_name, task_key) in enumerate(TASKS):
        train_mask = np.isfinite(train_labels[:, task_index])
        query_mask = np.isfinite(query_labels[:, task_index])
        train_task_embeddings = train_embeddings[train_mask]
        statuses = []
        for embedding in query_embeddings[query_mask]:
            distances = np.linalg.norm(train_task_embeddings - embedding, axis=1)
            distance = float(
                np.partition(distances, NEIGHBOR_COUNT - 1)[:NEIGHBOR_COUNT].mean()
            )
            statuses.append(_zone_from_cdf(float(task_kdes[task_index].cdf(distance))))
        count = max(len(statuses), 1)
        summaries.append(
            {
                "task": task_key,
                "display_name": display_name,
                "labeled_validation_count": int(query_mask.sum()),
                "inside_fraction": statuses.count("Inside") / count,
                "borderline_fraction": statuses.count("Borderline") / count,
                "outside_fraction": statuses.count("Outside") / count,
            }
        )
    return summaries


def main():
    args = parse_args()
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    task_keys = [key for _, key in TASKS]
    missing = [key for key in task_keys if key not in train.columns]
    if missing:
        raise ValueError(f"Training table is missing tasks: {missing}")

    model, _, device = load_predictor("cpu")
    train_smiles, train_embeddings = prepare_structures(
        train, model, device, "train"
    )
    _, validation_embeddings = prepare_structures(
        validation, model, device, "validation"
    )
    train_labels = train[task_keys].to_numpy(dtype=np.float32)
    validation_labels = validation[task_keys].to_numpy(dtype=np.float32)

    embedding_mean = train_embeddings.mean(axis=0).astype(np.float32)
    embedding_scale = train_embeddings.std(axis=0).astype(np.float32)
    embedding_scale[embedding_scale < 1e-6] = 1.0
    train_references = normalize_embeddings(
        train_embeddings, embedding_mean, embedding_scale
    )
    validation_references = normalize_embeddings(
        validation_embeddings, embedding_mean, embedding_scale
    )

    maximum_task_count = max(np.isfinite(train_labels).sum(axis=0))
    distance_references = np.full(
        (len(TASKS), maximum_task_count), np.nan, dtype=np.float32
    )
    task_counts = []
    task_kdes = []
    for task_index, (_, task_key) in enumerate(TASKS):
        indices = np.flatnonzero(np.isfinite(train_labels[:, task_index]))
        if len(indices) <= NEIGHBOR_COUNT:
            raise ValueError(f"Not enough labeled training data for {task_key}.")
        print(f"Calibrating {task_key} ({len(indices)} references)...", flush=True)
        distances = embedding_loo_distances(train_references, indices)
        distance_references[task_index, : len(distances)] = distances
        task_kdes.append(GaussianDistanceKDE(distances))
        task_counts.append(int(len(indices)))

    print("Calibrating global Gaussian-KDE coverage...", flush=True)
    global_embedding = embedding_loo_distances(
        train_references, np.arange(len(train), dtype=int)
    )
    global_kde = GaussianDistanceKDE(global_embedding)
    validation_summary = query_distance_summary(
        validation_references,
        validation_labels,
        train_references,
        train_labels,
        task_kdes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=train["ID"].astype(str).to_numpy(dtype=str),
        smiles=train_smiles,
        labels=train_labels,
        embedding_references=train_references,
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
        embedding_reference_distances=distance_references,
        global_embedding_reference=global_embedding,
    )
    metadata = {
        "artifact_version": 4,
        "domain_schema": EMBEDDING_DOMAIN_SCHEMA,
        "method": (
            "reflection-corrected Gaussian KDE over task-specific leave-one-out "
            "mean 5-NN Euclidean distances in standardized and L2-normalized "
            "final EEGNN embeddings"
        ),
        "embedding_source": (
            "one deterministic final embedding per training molecule, extracted "
            "immediately before the prediction layer from the selected checkpoint"
        ),
        "task_keys": task_keys,
        "task_names": [name for name, _ in TASKS],
        "task_reference_counts": task_counts,
        "neighbor_count": NEIGHBOR_COUNT,
        "embedding_original_dimensions": int(model.embedding_dim),
        "embedding_decision_dimensions": int(model.embedding_dim),
        "kde_kernel": "gaussian",
        "kde_boundary_correction": "reflection at zero",
        "kde_bandwidth_rule": "Scott",
        "inside_cdf_probability": INSIDE_CDF_PROBABILITY,
        "outer_cdf_probability": OUTER_CDF_PROBABILITY,
        "task_kde_bandwidths": [kde.bandwidth for kde in task_kdes],
        "task_inside_boundaries": [kde.inside_boundary for kde in task_kdes],
        "task_outer_boundaries": [kde.outer_boundary for kde in task_kdes],
        "global_kde_bandwidth": global_kde.bandwidth,
        "global_inside_boundary": global_kde.inside_boundary,
        "global_outer_boundary": global_kde.outer_boundary,
        "validation_summary": validation_summary,
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
    }
    with args.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {args.output}", flush=True)
    print(f"Saved {args.metadata}", flush=True)


if __name__ == "__main__":
    main()
