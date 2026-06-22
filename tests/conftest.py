"""
conftest.py — Shared fixtures for the GISTnet-MD test suite.

Every test file in this directory imports these fixtures automatically.
CPU-only execution is enforced globally via the environment variable below.
"""

import os

# ── Force CPU-only execution before any torch import ──────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import pytest
import torch
from torch_geometric.data import Data, Batch

# ── Make src/ importable without installing the package ───────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# =============================================================================
# SYNTHETIC GRAPH HELPERS
# =============================================================================

def make_single_graph(n_nodes: int = 10, cutoff: float = 10.0, seed: int = 0) -> Data:
    """
    Build a minimal synthetic PyG Data object that mimics a preprocessed frame.

    Attributes
    ----------
    x            : node type indices  (long)  – range [0, 20]
    pos          : 3-D coordinates   (float) – in Ångströms
    edge_index   : COO edge indices  (long)
    edge_attr    : pairwise distances (float)
    cutoff       : scalar metadata
    representation : string metadata
    """
    torch.manual_seed(seed)
    pos = torch.rand(n_nodes, 3) * 20.0           # random positions in a 20 Å box
    x   = torch.randint(0, 20, (n_nodes,))        # amino-acid type indices

    # Build a simple graph: connect every node to its two nearest neighbours
    # (just enough to give non-trivial edge_index without scipy/radius_graph)
    src, dst = [], []
    for i in range(n_nodes):
        dists_i = ((pos - pos[i]).pow(2).sum(-1)).sqrt()
        dists_i[i] = float("inf")                  # exclude self-loop
        neighbours = dists_i.argsort()[:4]         # 4 nearest neighbours
        for j in neighbours.tolist():
            src.append(i); dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    row, col   = edge_index
    edge_attr  = (pos[row] - pos[col]).norm(dim=-1)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        frame_index=0,
    )
    data.cutoff         = cutoff
    data.representation = "ca"
    data.node_dihe      = "none"
    return data


def make_batch_of_windows(
    n_logical_graphs: int = 2,
    window_size: int      = 4,
    n_nodes: int          = 8,
    cutoff: float         = 10.0,
) -> Batch:
    """
    Create a PyG Batch that simulates the output of ``collate_windows``:
    ``n_logical_graphs * window_size`` individual frames stacked into one Batch.

    Parameters
    ----------
    n_logical_graphs : number of MD windows in this mini-batch
    window_size      : number of frames per window
    n_nodes          : nodes per frame
    cutoff           : edge cutoff to embed as metadata
    """
    graphs = []
    for g in range(n_logical_graphs):
        for t in range(window_size):
            seed = g * window_size + t
            graphs.append(make_single_graph(n_nodes=n_nodes, cutoff=cutoff, seed=seed))
    return Batch.from_data_list(graphs)


# =============================================================================
# DISK I/O HELPERS  (shared across test_dataset and test_pipeline)
# =============================================================================

def write_pt_frames(directory: str, prefix: str, n_frames: int, n_nodes: int = 8):
    """
    Save *n_frames* synthetic PyG Data objects as ``<prefix>_frame_XXXXXX.pt``
    files inside *directory*.  Reusable across test modules.
    """
    import os as _os
    _os.makedirs(directory, exist_ok=True)
    for i in range(n_frames):
        data = make_single_graph(n_nodes=n_nodes, seed=i)
        data.frame_index = i
        fname = f"{prefix}_frame_{i:06d}.pt"
        torch.save(data, _os.path.join(directory, fname))


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def tiny_batch() -> Batch:
    """
    A minimal PyG Batch (2 windows × 4 frames × 8 nodes) for fast forward tests.
    Session-scoped so it is constructed only once per test run.
    """
    return make_batch_of_windows(n_logical_graphs=2, window_size=4, n_nodes=8)


@pytest.fixture(scope="session")
def small_embeddings():
    """
    Pre-computed random L2-normalised embeddings (8 samples, dim=16).
    Returned together with consistent labels and group ids.
    """
    torch.manual_seed(42)
    B, D = 8, 16
    emb    = torch.randn(B, D)
    emb    = torch.nn.functional.normalize(emb, p=2, dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    shared = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
    return emb, labels, groups, shared
