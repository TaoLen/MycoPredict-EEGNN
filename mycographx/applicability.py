"""Gaussian-KDE applicability domain over final EEGNN embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


NEIGHBOR_COUNT = 5
EMBEDDING_DOMAIN_SCHEMA = "embedding-kde-v4"
INSIDE_CDF_PROBABILITY = 0.95
OUTER_CDF_PROBABILITY = 0.99
KDE_CURVE_POINTS = 320
_SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def _mean_top_k(values, count=NEIGHBOR_COUNT):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return float("nan")
    count = min(int(count), values.size)
    return float(np.partition(values, count - 1)[:count].mean())


def _zone_from_cdf(cdf_probability):
    if not np.isfinite(cdf_probability):
        return "Unavailable"
    if cdf_probability > OUTER_CDF_PROBABILITY:
        return "Outside"
    if cdf_probability > INSIDE_CDF_PROBABILITY:
        return "Borderline"
    return "Inside"


def _scott_bandwidth(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    standard_deviation = float(values.std(ddof=1))
    data_range = float(np.ptp(values))
    fallback_scale = max(data_range / 6.0, abs(float(values.mean())) * 1e-3, 1e-6)
    scale = standard_deviation if standard_deviation > 1e-12 else fallback_scale
    return max(float(scale * values.size ** (-1.0 / 5.0)), 1e-6)


class GaussianDistanceKDE:
    """Reflection-corrected Gaussian KDE for non-negative distance values."""

    def __init__(self, reference_distances):
        reference = np.asarray(reference_distances, dtype=np.float64).reshape(-1)
        self.reference = np.sort(reference[np.isfinite(reference)])
        if self.reference.size < 3:
            raise ValueError("Gaussian KDE requires at least three distances.")
        if np.any(self.reference < 0):
            raise ValueError("Embedding distances must be non-negative.")
        self.bandwidth = _scott_bandwidth(self.reference)
        self.inside_boundary = self.quantile(INSIDE_CDF_PROBABILITY)
        self.outer_boundary = self.quantile(OUTER_CDF_PROBABILITY)

    def density(self, values):
        values = np.asarray(values, dtype=np.float64)
        original_shape = values.shape
        flat = values.reshape(-1)
        result = np.zeros(flat.size, dtype=np.float64)
        valid = np.isfinite(flat) & (flat >= 0)
        for start in range(0, int(valid.sum()), 512):
            valid_values = flat[valid][start : start + 512]
            differences = (
                valid_values[:, np.newaxis] - self.reference[np.newaxis, :]
            ) / self.bandwidth
            reflections = (
                valid_values[:, np.newaxis] + self.reference[np.newaxis, :]
            ) / self.bandwidth
            kernels = np.exp(-0.5 * differences**2) + np.exp(
                -0.5 * reflections**2
            )
            result[np.flatnonzero(valid)[start : start + 512]] = (
                kernels.mean(axis=1) / (self.bandwidth * _SQRT_TWO_PI)
            )
        return result.reshape(original_shape)

    def cdf(self, values):
        values = np.asarray(values, dtype=np.float64)
        original_shape = values.shape
        flat = values.reshape(-1)
        result = np.zeros(flat.size, dtype=np.float64)
        valid = np.isfinite(flat) & (flat >= 0)
        for start in range(0, int(valid.sum()), 512):
            valid_values = flat[valid][start : start + 512]
            upper = (
                valid_values[:, np.newaxis] - self.reference[np.newaxis, :]
            ) / self.bandwidth
            reflected = (
                valid_values[:, np.newaxis] + self.reference[np.newaxis, :]
            ) / self.bandwidth
            probabilities = ndtr(upper) + ndtr(reflected) - 1.0
            result[np.flatnonzero(valid)[start : start + 512]] = probabilities.mean(
                axis=1
            )
        result[np.isnan(flat)] = np.nan
        return np.clip(result.reshape(original_shape), 0.0, 1.0)

    def quantile(self, probability):
        probability = float(probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("KDE quantile probability must be between zero and one.")
        lower = 0.0
        upper = float(self.reference[-1] + 8.0 * self.bandwidth)
        while float(self.cdf(upper)) < probability:
            upper *= 2.0
        for _ in range(60):
            midpoint = (lower + upper) / 2.0
            if float(self.cdf(midpoint)) < probability:
                lower = midpoint
            else:
                upper = midpoint
        return float((lower + upper) / 2.0)

    def histogram_frame(self):
        q25, q75 = np.quantile(self.reference, [0.25, 0.75])
        width = 2.0 * float(q75 - q25) * self.reference.size ** (-1.0 / 3.0)
        if width > 1e-12:
            bin_count = int(np.ceil(np.ptp(self.reference) / width))
        else:
            bin_count = int(np.ceil(np.sqrt(self.reference.size)))
        bin_count = int(np.clip(bin_count, 15, 50))
        counts, edges = np.histogram(self.reference, bins=bin_count, density=False)
        widths = np.diff(edges)
        densities = counts / (self.reference.size * widths)
        return pd.DataFrame(
            {
                "Distance lower": edges[:-1],
                "Distance upper": edges[1:],
                "Histogram density": densities,
                "Reference count": counts,
            }
        )

    def curve_frame(self, query_distance, point_count=KDE_CURVE_POINTS):
        query_distance = float(query_distance)
        upper = max(
            float(self.reference[-1] + 3.0 * self.bandwidth),
            float(self.outer_boundary + 3.0 * self.bandwidth),
            query_distance * 1.05,
        )
        distances = np.linspace(0.0, upper, int(point_count), dtype=np.float64)
        return pd.DataFrame(
            {
                "Embedding distance": distances,
                "Gaussian KDE density": self.density(distances),
            }
        )


@dataclass
class EmbeddingApplicabilityAssessment:
    schema: str
    kde_statuses: np.ndarray
    embedding_distances: np.ndarray
    kde_percentiles: np.ndarray
    kde_densities: np.ndarray
    labeled_reference_counts: np.ndarray
    global_kde_status: str
    global_embedding_distance: float
    global_kde_percentile: float
    global_kde_density: float
    query_embedding_reference: np.ndarray
    all_embedding_distances: np.ndarray


@dataclass(frozen=True)
class EmbeddingKDEDistribution:
    histogram: pd.DataFrame
    curve: pd.DataFrame
    boundaries: pd.DataFrame
    query: pd.DataFrame
    bandwidth: float
    reference_count: int


class EmbeddingApplicabilityDomain:
    def __init__(self, artifact_path, metadata_path, model_path=None):
        artifact_path = Path(artifact_path)
        metadata_path = Path(metadata_path)
        if not artifact_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Applicability-domain artifacts are missing.")
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if int(self.metadata.get("artifact_version", 0)) != 4:
            raise ValueError(
                "Unsupported applicability-domain artifact version. Regenerate the "
                "embedding KDE artifacts with scripts/build_applicability_artifacts.py."
            )
        if self.metadata.get("domain_schema") != EMBEDDING_DOMAIN_SCHEMA:
            raise ValueError("The artifact is not an embedding KDE domain artifact.")
        if model_path is not None:
            digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            if digest != self.metadata.get("model_sha256"):
                raise ValueError("Applicability artifacts do not match the model checkpoint.")

        payload = np.load(artifact_path, allow_pickle=False)
        self.ids = payload["ids"]
        self.smiles = payload["smiles"]
        self.labels = payload["labels"].astype(np.float32, copy=False)
        self.embedding_references = payload["embedding_references"].astype(
            np.float32, copy=False
        )
        self.embedding_mean = payload["embedding_mean"].astype(np.float32, copy=False)
        self.embedding_scale = payload["embedding_scale"].astype(np.float32, copy=False)
        self.embedding_reference_distances = payload[
            "embedding_reference_distances"
        ].astype(np.float32, copy=False)
        self.global_embedding_reference = payload[
            "global_embedding_reference"
        ].astype(np.float32, copy=False)
        self.task_keys = tuple(self.metadata["task_keys"])
        self.task_names = tuple(self.metadata["task_names"])
        self.task_masks = np.isfinite(self.labels)
        self._validate()
        self.task_kdes = tuple(
            GaussianDistanceKDE(self.embedding_reference_distances[index])
            for index in range(len(self.task_keys))
        )
        self.global_kde = GaussianDistanceKDE(self.global_embedding_reference)

    def _validate(self):
        row_count = len(self.smiles)
        task_count = len(self.task_keys)
        embedding_dimensions = self.embedding_mean.size
        if self.labels.shape != (row_count, task_count):
            raise ValueError("Applicability labels have an incompatible shape.")
        if self.embedding_references.shape != (row_count, embedding_dimensions):
            raise ValueError("Applicability embeddings have an incompatible shape.")
        if self.embedding_scale.shape != self.embedding_mean.shape:
            raise ValueError("Embedding normalization has an incompatible shape.")
        if self.embedding_reference_distances.shape[0] != task_count:
            raise ValueError("KDE distance references have an incompatible shape.")
        if np.isfinite(self.embedding_reference_distances).sum(axis=1).tolist() != (
            self.task_masks.sum(axis=0).tolist()
        ):
            raise ValueError("KDE distance counts do not match labeled references.")

    def _transform_embedding(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != self.embedding_mean.shape:
            raise ValueError("Prediction embedding is incompatible with AD artifacts.")
        standardized = (embedding - self.embedding_mean) / self.embedding_scale
        norm = float(np.linalg.norm(standardized))
        normalized = standardized / norm if norm > 1e-12 else standardized
        return normalized.astype(np.float32)

    def assess(self, prediction):
        normalized_embedding = self._transform_embedding(prediction.embedding)
        all_distances = np.linalg.norm(
            self.embedding_references - normalized_embedding.reshape(1, -1), axis=1
        ).astype(np.float32)

        task_count = len(self.task_keys)
        distances = np.zeros(task_count, dtype=np.float32)
        percentiles = np.zeros(task_count, dtype=np.float32)
        densities = np.zeros(task_count, dtype=np.float32)
        zones = np.empty(task_count, dtype="<U12")
        for task_index, kde in enumerate(self.task_kdes):
            distance = _mean_top_k(all_distances[self.task_masks[:, task_index]])
            cdf_probability = float(kde.cdf(distance))
            distances[task_index] = distance
            percentiles[task_index] = 100.0 * cdf_probability
            densities[task_index] = float(kde.density(distance))
            zones[task_index] = _zone_from_cdf(cdf_probability)

        global_distance = _mean_top_k(all_distances)
        global_cdf = float(self.global_kde.cdf(global_distance))
        return EmbeddingApplicabilityAssessment(
            schema=EMBEDDING_DOMAIN_SCHEMA,
            kde_statuses=zones,
            embedding_distances=distances,
            kde_percentiles=percentiles,
            kde_densities=densities,
            labeled_reference_counts=self.task_masks.sum(axis=0).astype(np.int32),
            global_kde_status=_zone_from_cdf(global_cdf),
            global_embedding_distance=global_distance,
            global_kde_percentile=100.0 * global_cdf,
            global_kde_density=float(self.global_kde.density(global_distance)),
            query_embedding_reference=normalized_embedding,
            all_embedding_distances=all_distances,
        )

    def task_frame(self, assessment):
        return pd.DataFrame(
            {
                "Target": self.task_names,
                "KDE domain status": assessment.kde_statuses,
                "Mean 5-NN embedding distance": assessment.embedding_distances,
                "KDE CDF percentile": assessment.kde_percentiles,
                "Gaussian KDE density": assessment.kde_densities,
                "Labeled embedding references": assessment.labeled_reference_counts,
            }
        )

    def neighbor_frame(self, assessment, task_index, count=NEIGHBOR_COUNT):
        task_index = int(task_index)
        candidates = np.flatnonzero(self.task_masks[:, task_index])
        distances = assessment.all_embedding_distances[candidates]
        count = min(int(count), len(candidates))
        selected_local = np.argpartition(distances, count - 1)[:count]
        selected = candidates[selected_local]
        selected = selected[np.argsort(assessment.all_embedding_distances[selected])]
        labels = self.labels[selected, task_index]
        return pd.DataFrame(
            {
                "Reference index": selected,
                "ID": self.ids[selected],
                "SMILES": self.smiles[selected],
                "Embedding distance": assessment.all_embedding_distances[selected],
                "Experimental class": np.where(labels >= 0.5, "Active", "Inactive"),
            }
        )

    def kde_distribution(self, assessment, task_index):
        task_index = int(task_index)
        kde = self.task_kdes[task_index]
        query_distance = float(assessment.embedding_distances[task_index])
        return EmbeddingKDEDistribution(
            histogram=kde.histogram_frame(),
            curve=kde.curve_frame(query_distance),
            boundaries=pd.DataFrame(
                {
                    "Boundary": ["95% KDE boundary", "99% KDE boundary"],
                    "Embedding distance": [
                        kde.inside_boundary,
                        kde.outer_boundary,
                    ],
                }
            ),
            query=pd.DataFrame(
                {
                    "Marker": ["Query molecule"],
                    "Embedding distance": [query_distance],
                    "Gaussian KDE density": [
                        float(assessment.kde_densities[task_index])
                    ],
                }
            ),
            bandwidth=kde.bandwidth,
            reference_count=kde.reference.size,
        )


def load_applicability_domain(artifact_path, metadata_path, model_path=None):
    return EmbeddingApplicabilityDomain(
        artifact_path,
        metadata_path,
        model_path=model_path,
    )
