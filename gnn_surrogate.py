"""Train and evaluate a graph neural network surrogate model for PROTAC activity."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

# The following imports require rdkit and torch_geometric which are not included
# in this repository by default. They are only used when actually training the
# model.
try:
    from rdkit import Chem
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception:  # pragma: no cover - modules may be missing when linting
    Chem = None
    Data = None
    GATConv = None
    def global_mean_pool(x, batch):  # type: ignore
        return x


class ProtacDataset(Dataset):
    """Simple dataset wrapping PROTAC graphs and protein sequences."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        entry = self.data[idx]
        return entry["graph"], entry["sequence"], entry["label"]


def smiles_to_graph(smiles: str) -> Data:
    """Convert a SMILES string into a PyG Data object."""
    if Chem is None:
        raise ImportError("rdkit is required for graph construction")
    mol = Chem.MolFromSmiles(smiles)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    edges = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index)


def sequence_to_tensor(seq: str, vocab: str = "ACDEFGHIKLMNPQRSTVWY") -> torch.Tensor:
    """Map an amino-acid sequence to integer indices."""
    mapping = {ch: i + 1 for i, ch in enumerate(vocab)}
    idx = [mapping.get(ch.upper(), 0) for ch in seq]
    return torch.tensor(idx, dtype=torch.long)


class SequenceEncoder(nn.Module):
    """Simple CNN-based encoder for protein sequences."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x


class GraphEncoder(nn.Module):
    """Graph Attention Network encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = GATConv(1, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        return global_mean_pool(x, data.batch)


class SurrogateModel(nn.Module):
    """Joint graph + sequence model."""

    def __init__(self, hidden_dim: int = 128, vocab_size: int = 20):
        super().__init__()
        self.graph_encoder = GraphEncoder(hidden_dim)
        self.seq_encoder = SequenceEncoder(vocab_size, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, 1)

    def forward(self, graph: Data, seq: torch.Tensor) -> torch.Tensor:
        g_emb = self.graph_encoder(graph)
        s_emb = self.seq_encoder(seq)
        h = torch.cat([g_emb, s_emb], dim=1)
        return self.classifier(h).squeeze(-1)


def train(model: nn.Module, loader: DataLoader, epochs: int = 10) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for graph, seq, label in loader:
            out = model(graph, seq)
            loss = loss_fn(out, label.float())
            loss.backward()
            opt.step()
            opt.zero_grad()


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for graph, seq, label in loader:
            out = torch.sigmoid(model(graph, seq))
            preds.extend(out.tolist())
            labels.extend(label.tolist())
    roc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, [p > 0.5 for p in preds])
    return {"roc_auc": roc, "f1": f1}


if __name__ == "__main__":  # pragma: no cover - example usage
    data_path = Path("protac_dataset.pkl")
    dataset = ProtacDataset(pickle.load(open(data_path, "rb")))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SurrogateModel()
    train(model, loader)
    metrics = evaluate(model, loader)
    print(metrics)
    torch.save(model.state_dict(), "gnn_surrogate.pt")
