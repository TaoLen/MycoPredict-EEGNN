"""EEGNN architecture and molecular graph featurization used by the app."""

from __future__ import annotations

import torch
import torch.nn as nn
from rdkit import Chem
from torch.distributions import Dirichlet, Gamma
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool


ATOMIC_NUMBERS = [
    1, 11, 19, 12, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    5, 13, 14, 15, 33, 6, 7, 8, 16, 34, 9, 17, 35, 53,
]
DEGREES = list(range(8))
CHARGES = [-1, 0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    "R",
    "S",
]

NODE_DIM = len(ATOMIC_NUMBERS) + len(DEGREES) + len(CHARGES) + len(HYBRIDIZATIONS) + 1
EDGE_DIM = 4 + 1 + 1 + len(BOND_STEREO)


def _one_hot(value, categories):
    index = categories.index(value)
    return [int(position == index) for position in range(len(categories))]


def _atom_features(atom):
    hybridization = atom.GetHybridization()
    if hybridization not in HYBRIDIZATIONS:
        hybridization = Chem.rdchem.HybridizationType.SP3
    values = (
        _one_hot(atom.GetAtomicNum(), ATOMIC_NUMBERS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), CHARGES)
        + _one_hot(hybridization, HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
    )
    return torch.tensor(values, dtype=torch.float32)


def _bond_features(bond):
    values = (
        _one_hot(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        )
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
    )
    return torch.tensor(values, dtype=torch.float32)


def smiles_to_graph(smiles: str) -> tuple[str, Chem.Mol, Data]:
    """Validate a SMILES and reproduce the training graph representation."""
    text = str(smiles).strip()
    if not text:
        raise ValueError("SMILES cannot be empty.")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError("Invalid SMILES or structure could not be sanitized.")
    if mol.GetNumAtoms() == 0:
        raise ValueError("The structure does not contain any atoms.")
    if mol.GetNumBonds() == 0:
        raise ValueError("The model requires a structure with at least one bond.")

    unsupported = sorted(
        {atom.GetAtomicNum() for atom in mol.GetAtoms()} - set(ATOMIC_NUMBERS)
    )
    if unsupported:
        symbols = [Chem.GetPeriodicTable().GetElementSymbol(number) for number in unsupported]
        raise ValueError(
            "Element(s) outside the model domain: " + ", ".join(symbols) + "."
        )

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atoms = torch.stack([_atom_features(atom) for atom in mol.GetAtoms()])
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_features.append(_bond_features(bond))

    graph = Data(
        x=atoms,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features),
    )
    return canonical, mol, graph


class SupervisedUncertainty(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = "classification"
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, predictions, targets, mask):
        losses = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )
        weights = self.log_vars.view(1, -1).to(losses.device)
        return ((torch.exp(-weights) * losses + weights) * mask.float()).sum() / mask.sum()


class DMPGM:
    """Dirichlet mixture generator retained exactly for checkpoint compatibility."""

    def __init__(
        self,
        num_nodes,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
        device,
    ):
        self.N = num_nodes
        self.K = num_components
        self.alpha = concentration
        self.prior_amp = prior_concentration
        self.mcmc_iters = mcmc_iters
        self.device = device
        self.u = torch.zeros(self.N + 1, device=device)
        self.wk = torch.ones(self.K + 1, self.N + 1, device=device)
        self.pi = torch.ones(self.K + 1, device=device) / (self.K + 1)
        self.c = None
        self.z = None
        self.hmc_stepsize = 0.1
        self.hmc_leapfrog_steps = 5

    def logp_u(self, u, counts_w0):
        w0 = u.exp()
        sw0 = w0.sum()
        return -sw0 + (counts_w0 * u).sum() - sw0

    def mcmc_update(self, edge_index):
        i, j = edge_index.cpu()
        indices = torch.stack([i, j], dim=1)

        for _ in range(self.mcmc_iters):
            scores = (
                self.pi.unsqueeze(1)
                * self.wk[:, indices[:, 0].to(self.device)]
                * self.wk[:, indices[:, 1].to(self.device)]
            )
            scores = torch.nan_to_num(
                scores, nan=0.0, posinf=1e6, neginf=0.0
            ).clamp(min=1e-8, max=1e6)
            self.c = torch.multinomial(scores.t(), num_samples=1).squeeze(1)

            rates = torch.nan_to_num(
                scores.sum(dim=0), nan=0.0, posinf=1e6, neginf=0.0
            ).clamp(min=1e-8, max=1e6).cpu()
            z_cpu = torch.poisson(rates)
            z_cpu[z_cpu == 0] = 1
            self.z = (indices, z_cpu.to(self.device))

            components = self.c.repeat(2)
            nodes = torch.cat([indices[:, 0], indices[:, 1]], dim=0)
            counts_values = z_cpu.repeat(2).to(self.device)
            counts = torch.zeros(self.K + 1, self.N + 1, device=self.device)
            counts.index_put_(
                (components, nodes.to(self.device)), counts_values, accumulate=True
            )

            shape = torch.exp(torch.clamp(self.u, min=-20.0, max=20.0)).unsqueeze(0)
            shape = torch.nan_to_num(
                shape + counts, nan=1.0, posinf=1e6, neginf=1e-6
            ).clamp(min=1e-6, max=1e6)
            self.wk = Gamma(shape, torch.ones_like(shape)).sample()

            component_counts = torch.bincount(
                self.c, minlength=self.K + 1
            ).float().to(self.device)
            component_counts[0] += self.alpha
            self.pi = Dirichlet(component_counts + self.prior_amp).sample()
            self.pi = self.pi / self.pi.sum()
            counts_w0 = counts[0]

            with torch.enable_grad():
                old_u = self.u.clone().detach().requires_grad_(True)
                old_p = torch.randn_like(old_u)
                potential = lambda value: -self.logp_u(value, counts_w0)
                old_h = potential(old_u) + 0.5 * (old_p**2).sum()
                momentum = old_p - 0.5 * self.hmc_stepsize * torch.autograd.grad(
                    potential(old_u), old_u
                )[0]
                new_u = old_u.clone()
                for _ in range(self.hmc_leapfrog_steps):
                    new_u = new_u + self.hmc_stepsize * momentum
                    momentum = momentum - self.hmc_stepsize * torch.autograd.grad(
                        potential(new_u), new_u
                    )[0]
                momentum = momentum - 0.5 * self.hmc_stepsize * torch.autograd.grad(
                    potential(new_u), new_u
                )[0]
                new_h = potential(new_u) + 0.5 * (momentum**2).sum()
                if torch.rand(1, device=self.device) < torch.exp(old_h - new_h):
                    self.u = torch.clamp(new_u.detach(), min=-20.0, max=20.0)

    def virtual_graph(self):
        indices, weights = self.z
        indices = indices.to(self.device)
        weights = weights.to(self.device)
        both_directions = torch.cat([indices, indices[:, [1, 0]]], dim=0)
        both_weights = torch.cat([weights, weights], dim=0)
        degree = torch.zeros(self.N, device=self.device)
        degree = degree.index_add(0, both_directions[:, 0], both_weights)
        inverse_sqrt = degree.pow(-0.5)
        inverse_sqrt[torch.isinf(inverse_sqrt)] = 0.0
        normalized = (
            both_weights
            * inverse_sqrt[both_directions[:, 0]]
            * inverse_sqrt[both_directions[:, 1]]
        )
        return both_directions.t(), normalized


class EEGNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
    ):
        super().__init__(aggr="add")
        self.generator_args = {
            "num_components": num_components,
            "concentration": concentration,
            "prior_concentration": prior_concentration,
            "mcmc_iters": mcmc_iters,
        }
        self.generator = None
        self.message_proj = nn.Sequential(
            nn.Linear(input_dim + edge_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.node_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.gru = nn.GRUCell(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        node_count = x.size(0)
        if self.generator is None or self.generator.N != node_count:
            self.generator = DMPGM(
                num_nodes=node_count, device=device, **self.generator_args
            )
        self.generator.mcmc_update(edge_index)
        virtual_edges, virtual_weights = self.generator.virtual_graph()

        original = {
            (source.item(), target.item()): edge_attr[index]
            for index, (source, target) in enumerate(edge_index.t())
        }
        empty = torch.zeros(edge_attr.size(1), device=device)
        features = [
            original.get((source, target), original.get((target, source), empty))
            for source, target in virtual_edges.t().tolist()
        ]
        virtual_attr = torch.cat(
            [torch.stack(features), virtual_weights.unsqueeze(-1)], dim=1
        )
        aggregated = self.propagate(
            virtual_edges, x=x, edge_attr=virtual_attr
        )
        return self.gru(aggregated, self.node_proj(x))

    def message(self, x_j, edge_attr):
        return self.message_proj(torch.cat([x_j, edge_attr], dim=-1))


def _activation(name):
    activations = {
        "relu": nn.ReLU,
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.2),
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]()


class EEGNNet(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation,
        dropout_rate,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
        num_tasks,
    ):
        super().__init__()
        self.uncertainty = SupervisedUncertainty(num_tasks)
        self.agg_layers = nn.ModuleList(
            [
                EEGNNLayer(
                    input_dim=node_dim if index == 0 else agg_hidden_dims[index - 1],
                    output_dim=agg_hidden_dims[index],
                    edge_dim=edge_dim,
                    dropout_rate=dropout_rate,
                    num_components=num_components,
                    concentration=concentration,
                    prior_concentration=prior_concentration,
                    mcmc_iters=mcmc_iters,
                )
                for index in range(num_agg_layers)
            ]
        )
        self.lin_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        agg_hidden_dims[-1] if index == 0 else lin_hidden_dims[index - 1],
                        lin_hidden_dims[index],
                    ),
                    _activation(activation),
                    nn.Dropout(dropout_rate),
                )
                for index in range(num_lin_layers)
            ]
        )
        self.embedding_dim = lin_hidden_dims[-1]
        self.embedding_layer = nn.Linear(lin_hidden_dims[-1], self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, num_tasks)

    def encode(self, data):
        """Return the molecular representation immediately before prediction."""
        x = data.x
        for layer in self.agg_layers:
            x = layer(x, data.edge_index, data.edge_attr)
        x = global_add_pool(x, data.batch)
        for layer in self.lin_layers:
            x = layer(x)
        return self.embedding_layer(x)

    def forward(self, data, return_embedding=False):
        embedding = self.encode(data)
        if return_embedding:
            return embedding
        return self.output_layer(embedding)
