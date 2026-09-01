"""Counterfactual atom-contribution maps for the EEGNN predictor."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from torch_geometric.data import Batch

from .augmentations import apply_rulebook_perturbations
from .rules import feature_slices


COUNTERFACTUAL_FAMILIES = [
    "SP2_POLAR_ALL",
    "SP3_POLAR_ALL",
    "SP2_APOLAR_ALL",
    "SP3_APOLAR_ALL",
    "SP2_REACTIVE_ALL",
    "SP3_REACTIVE_ALL",
    "REDOX_FAMILY",
    "ACYL_FAMILY_ALL",
    "CARBAMATE_FAMILY_ALL",
    "AMIDE_FAMILY_ALL",
    "SULFURE_FAMILY_ALL",
    "PHOSPHORUS_FAMILY_ALL",
    "TOGGLE_CHARGE_FAMILY_ALL",
    "POLYVALENT_FAMILY_ALL",
    "ALIPHATIC_FAMILY_ALL",
    "TOGGLE_RING_FAMILY_ALL",
    "RING_FAMILY_ALL",
    "BOND_FAMILY_ALL",
    "DIARYL_FAMILY_ALL",
]
RULEBOOK_VERSION = "2.0.0"


@dataclass
class Explanation:
    task_index: int
    base_probability: float
    raw_contributions: np.ndarray
    normalized_contributions: np.ndarray
    relative_contributions: np.ndarray
    standard_deviations: np.ndarray
    transformations_per_atom: np.ndarray
    generated_transformations: int
    evaluated_transformations: int
    failed_transformations: int
    unchanged_transformations: int
    atom_table: pd.DataFrame
    transformation_table: pd.DataFrame
    png: bytes
    svg: bytes
    relative_png: bytes
    relative_svg: bytes


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _stable_seed(canonical_smiles):
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31 - 1)


def _reset_generators(model):
    for layer in model.agg_layers:
        layer.generator = None


def _predict_graph(model, graph, device, seed):
    _seed_everything(seed)
    _reset_generators(model)
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        logits = model(batch)
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def _rule_family(rule_id):
    parts = str(rule_id).split("_")
    if "ALL" in parts:
        index = parts.index("ALL")
        return "_".join(parts[: index + 1])
    if "FAMILY" in parts:
        index = parts.index("FAMILY")
        return "_".join(parts[: index + 1])
    return parts[0]


def _adjacency(edge_index, node_count):
    result = [set() for _ in range(node_count)]
    for source, target in edge_index.t().tolist():
        result[source].add(target)
        result[target].add(source)
    return result


def _changed_group(raw, candidate, reported_nodes, focus_index):
    original_count = raw.x.size(0)
    candidate_count = candidate.x.size(0)
    group = {int(index) for index in (reported_nodes or []) if index < original_count}
    if group:
        return sorted(group)

    slices = feature_slices()
    atomic_start, atomic_end = slices["atomic"]
    shared_count = min(original_count, candidate_count)
    for index in range(shared_count):
        if not torch.equal(
            raw.x[index, atomic_start:atomic_end],
            candidate.x[index, atomic_start:atomic_end],
        ):
            group.add(index)

    if original_count == candidate_count:
        original_adj = _adjacency(raw.edge_index, original_count)
        candidate_adj = _adjacency(candidate.edge_index, candidate_count)
        for index in range(original_count):
            if original_adj[index] != candidate_adj[index]:
                group.add(index)

    return sorted(group or {focus_index})


def _same_graph(first, second):
    return (
        torch.equal(first.x, second.x)
        and torch.equal(first.edge_index, second.edge_index)
        and torch.equal(first.edge_attr, second.edge_attr)
    )


def _render_map(molecule, weights):
    mol = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    values = weights.tolist()
    color_map = [
        (0.18, 0.38, 0.72),
        (1.0, 1.0, 1.0),
        (0.74, 0.24, 0.21),
    ]

    png_drawer = rdMolDraw2D.MolDraw2DCairo(900, 620)
    png_drawer.drawOptions().useBWAtomPalette()
    png_drawer.drawOptions().addAtomIndices = True
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        values,
        colorMap=color_map,
        scale=1.0,
        alpha=0.48,
        contourLines=8,
        draw2d=png_drawer,
    )
    png_drawer.FinishDrawing()

    svg_drawer = rdMolDraw2D.MolDraw2DSVG(900, 620)
    svg_drawer.drawOptions().useBWAtomPalette()
    svg_drawer.drawOptions().addAtomIndices = True
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        values,
        colorMap=color_map,
        scale=1.0,
        alpha=0.48,
        contourLines=8,
        draw2d=svg_drawer,
    )
    svg_drawer.FinishDrawing()
    return (
        png_drawer.GetDrawingText(),
        svg_drawer.GetDrawingText().encode("utf-8"),
    )


def _relative_contrast(contributions):
    """Center mapped atom effects as in the original explanation pipeline."""
    relative = np.zeros_like(contributions, dtype=float)
    mapped = np.abs(contributions) > 1e-12
    if int(mapped.sum()) < 2:
        return relative

    values = contributions[mapped]
    deviation = float(values.std(ddof=0))
    if deviation <= 1e-12:
        return relative

    relative[mapped] = (values - float(values.mean())) / (deviation + 1e-8)
    scale = float(np.max(np.abs(relative)))
    if scale > 1e-12:
        relative /= scale
    return relative


def explain_prediction(
    model,
    device,
    graph,
    molecule,
    canonical_smiles,
    task_index,
):
    """Evaluate every valid rulebook perturbation over the whole molecule."""
    if not 0 <= int(task_index) < 9:
        raise ValueError("Invalid target index.")

    raw = copy.deepcopy(graph).cpu()
    seed = _stable_seed(canonical_smiles)
    base_probabilities = _predict_graph(model, raw, device, seed)
    base_probability = float(base_probabilities[int(task_index)])
    node_count = raw.x.size(0)
    per_atom = [[] for _ in range(node_count)]
    transformation_rows = []
    generated = 0
    failed = 0
    unchanged = 0

    for atom_index in range(node_count):
        candidates = apply_rulebook_perturbations(
            raw,
            atom_index,
            families=COUNTERFACTUAL_FAMILIES,
        )
        generated += len(candidates)
        for candidate, reported_nodes, rule_id in candidates:
            if _same_graph(raw, candidate):
                unchanged += 1
                continue
            if (
                candidate.x.numel() == 0
                or candidate.edge_index.numel() == 0
                or candidate.edge_attr.numel() == 0
            ):
                failed += 1
                continue
            try:
                probabilities = _predict_graph(model, candidate, device, seed)
            except (RuntimeError, ValueError, IndexError):
                failed += 1
                continue

            changed_nodes = _changed_group(
                raw, candidate, reported_nodes, atom_index
            )
            delta = base_probability - float(probabilities[int(task_index)])
            share = delta / max(len(changed_nodes), 1)
            for changed_index in changed_nodes:
                if changed_index < node_count:
                    per_atom[changed_index].append(share)
            transformation_rows.append(
                {
                    "Rule": str(rule_id),
                    "Family": _rule_family(rule_id),
                    "Focus atom": atom_index,
                    "Affected atoms": ", ".join(
                        str(index) for index in changed_nodes
                    ),
                    "Counterfactual score": float(
                        probabilities[int(task_index)]
                    ),
                    "Delta": delta,
                }
            )

    means = np.zeros(node_count, dtype=float)
    deviations = np.zeros(node_count, dtype=float)
    counts = np.zeros(node_count, dtype=int)
    consistency = np.zeros(node_count, dtype=float)
    for atom_index, values in enumerate(per_atom):
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        means[atom_index] = float(array.mean())
        deviations[atom_index] = float(array.std(ddof=0))
        counts[atom_index] = int(array.size)
        consistency[atom_index] = abs(
            float(np.mean(array > 0) - np.mean(array < 0))
        )

    contributions = means * consistency
    scale = float(np.max(np.abs(contributions)))
    normalized = contributions / scale if scale > 1e-12 else contributions.copy()
    relative = _relative_contrast(contributions)
    png, svg = _render_map(molecule, normalized)
    relative_png, relative_svg = _render_map(molecule, relative)

    atom_rows = []
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        atom_rows.append(
            {
                "Atom": atom_index,
                "Element": atom.GetSymbol(),
                "Score effect": float(contributions[atom_index]),
                "Normalized effect": float(normalized[atom_index]),
                "Relative contrast": float(relative[atom_index]),
                "Std dev": float(deviations[atom_index]),
                "Transformations": int(counts[atom_index]),
            }
        )

    transformations = pd.DataFrame(transformation_rows)
    if not transformations.empty:
        transformations = transformations.sort_values(
            "Delta", key=lambda values: values.abs(), ascending=False
        ).reset_index(drop=True)

    return Explanation(
        task_index=int(task_index),
        base_probability=base_probability,
        raw_contributions=contributions,
        normalized_contributions=normalized,
        relative_contributions=relative,
        standard_deviations=deviations,
        transformations_per_atom=counts,
        generated_transformations=generated,
        evaluated_transformations=len(transformation_rows),
        failed_transformations=failed,
        unchanged_transformations=unchanged,
        atom_table=pd.DataFrame(atom_rows),
        transformation_table=transformations,
        png=png,
        svg=svg,
        relative_png=relative_png,
        relative_svg=relative_svg,
    )
