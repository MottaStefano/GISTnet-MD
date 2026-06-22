"""
test_pipeline.py — Integration smoke test for the full training loop.

Target: verifies that a complete forward→loss→backward→optimizer.step() cycle
runs without errors, OOM, or NaN values.  Uses ultra-lightweight hyper-
parameters and synthetic .pt files; no GPU, no real trajectory data needed.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import pytest
import torch
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.dataset     import MDFlexibleWindowDataset, collate_windows
from core.architecture import HybridStSchnet
from core.loss         import GroupContrastiveLoss
from conftest          import make_single_graph


# =============================================================================
# HELPERS
# =============================================================================

def _write_pt_frames(directory: str, prefix: str, n_frames: int, n_nodes: int = 8):
    """Save synthetic .pt files that mimic preprocessed MD frames."""
    os.makedirs(directory, exist_ok=True)
    for i in range(n_frames):
        data = make_single_graph(n_nodes=n_nodes, seed=i)
        data.frame_index = i
        fname = f"{prefix}_frame_{i:06d}.pt"
        torch.save(data, os.path.join(directory, fname))


# =============================================================================
# SMOKE TEST FIXTURES
# =============================================================================

WINDOW_SIZE   = 4   # frames per MD window (kept tiny for speed)
HIDDEN_DIM    = 8   # smallest useful hidden dim
EMBEDDING_DIM = 4
N_LAYERS      = 1
N_FRAMES      = 12  # frames per class/group  →  (12-4)//4+1 = 3 windows each
N_NODES       = 8   # nodes per frame


@pytest.fixture(scope="module")
def synthetic_data_dirs(tmp_path_factory):
    """
    Create two class directories with synthetic .pt files.
    Module-scoped so files are written only once for all tests in this module.
    """
    root = tmp_path_factory.mktemp("pipeline_data")
    for cls in range(2):
        for grp in range(2):
            d = root / f"class_{cls}" / f"rep_{grp}"
            _write_pt_frames(str(d), prefix=f"sim_c{cls}g{grp}", n_frames=N_FRAMES)
    return root


@pytest.fixture(scope="module")
def dataset_and_loader(synthetic_data_dirs):
    """
    Build MDFlexibleWindowDataset + DataLoader from the synthetic files.
    Module-scoped so the dataset is constructed only once.
    """
    class_dirs = [
        str(synthetic_data_dirs / "class_0"),
        str(synthetic_data_dirs / "class_1"),
    ]
    ds = MDFlexibleWindowDataset(
        data_class_dirs=class_dirs,
        split="train",
        window_size=WINDOW_SIZE,
        window_offset=WINDOW_SIZE,   # non-overlapping windows
        stride=1,
        ignore_validation_logic=True,
    )
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=0,   # single process for test portability
    )
    return ds, loader


@pytest.fixture(scope="module")
def tiny_model():
    """Construct and return a tiny HybridStSchnet in training mode."""
    model = HybridStSchnet(
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        window_size=WINDOW_SIZE,
        n_layers=N_LAYERS,
        cutoff=10.0,
        temporal_setup="cnn",
        pooling_activation="softmax",
    )
    model.train()
    return model


# =============================================================================
# PIPELINE TESTS
# =============================================================================

class TestFastTrainingLoop:
    """Integration smoke tests for one mini-batch of training."""

    def test_dataset_not_empty(self, dataset_and_loader):
        ds, _ = dataset_and_loader
        assert len(ds) > 0, "Synthetic dataset is empty — check file creation."

    def test_loader_yields_batches(self, dataset_and_loader):
        _, loader = dataset_and_loader
        batch_iter = iter(loader)
        batch = next(batch_iter, None)
        assert batch is not None, "DataLoader yielded no batches."

    def test_batch_structure(self, dataset_and_loader):
        """Each batch must be a 5-tuple with the expected types."""
        from torch_geometric.data import Batch
        _, loader = dataset_and_loader
        pyg_batch, labels, groups, shared, paths = next(iter(loader))

        assert isinstance(pyg_batch, Batch), "First element must be a PyG Batch."
        assert labels.dim()  == 1, "Labels must be 1-D."
        assert groups.dim()  == 1, "Groups must be 1-D."
        assert isinstance(paths, list), "Paths must be a list."

    def test_forward_pass_no_crash(self, dataset_and_loader, tiny_model):
        """Model.forward() must complete without raising any exception."""
        _, loader = dataset_and_loader
        pyg_batch, *_ = next(iter(loader))
        out = tiny_model(pyg_batch)
        assert out is not None

    def test_forward_output_shape(self, dataset_and_loader, tiny_model):
        """Output shape must be (logical_batch_size, embedding_dim)."""
        _, loader = dataset_and_loader
        pyg_batch, labels, *_ = next(iter(loader))
        out = tiny_model(pyg_batch)
        # logical batch = number of windows in this mini-batch
        expected_batch = labels.shape[0]
        assert out.shape == (expected_batch, EMBEDDING_DIM), (
            f"Expected ({expected_batch}, {EMBEDDING_DIM}), got {out.shape}"
        )

    def test_forward_no_nan(self, dataset_and_loader, tiny_model):
        _, loader = dataset_and_loader
        pyg_batch, *_ = next(iter(loader))
        out = tiny_model(pyg_batch)
        assert not torch.isnan(out).any(), "Forward pass produced NaN outputs."

    def test_loss_computation(self, dataset_and_loader, tiny_model):
        """GroupContrastiveLoss must produce a finite scalar from model output."""
        _, loader = dataset_and_loader
        pyg_batch, labels, groups, shared, _ = next(iter(loader))

        out     = tiny_model(pyg_batch)
        loss_fn = GroupContrastiveLoss(margin=1.0, mode="group_aware")
        loss    = loss_fn(out, labels, groups, shared)

        assert not torch.isnan(loss), "Loss is NaN."
        assert not torch.isinf(loss), "Loss is Inf."
        assert loss.item() >= 0.0,    "Loss is negative."

    def test_backward_and_optimizer_step(self, dataset_and_loader, tiny_model):
        """
        The full backward pass + optimizer.step() must complete successfully
        and leave no NaN gradients.
        """
        _, loader = dataset_and_loader
        pyg_batch, labels, groups, shared, _ = next(iter(loader))

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        out     = tiny_model(pyg_batch)
        loss_fn = GroupContrastiveLoss(margin=1.0, mode="group_aware")
        loss    = loss_fn(out, labels, groups, shared)
        loss.backward()

        # Check no NaN in any gradient
        for name, param in tiny_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient detected in parameter '{name}'."
                )

        optimizer.step()   # Must not raise

    def test_embedding_weight_gradient_exists(self, dataset_and_loader, tiny_model):
        """
        After backward, the embedding layer's weight.grad must be populated,
        confirming the full computational graph is intact.
        """
        _, loader = dataset_and_loader
        pyg_batch, labels, groups, shared, _ = next(iter(loader))

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        out     = tiny_model(pyg_batch)
        loss_fn = GroupContrastiveLoss(margin=1.0, mode="group_aware")
        loss    = loss_fn(out, labels, groups, shared)
        loss.backward()

        grad = tiny_model.spatial_encoder.embedding.weight.grad
        assert grad is not None, (
            "embedding.weight.grad is None — computational graph may be detached."
        )

    def test_multiple_batches_no_crash(self, dataset_and_loader, tiny_model):
        """
        Iterate over all available batches in the loader without any exception.
        This catches issues that only surface after the first batch (e.g.,
        BatchNorm statistics with variable batch sizes).
        """
        _, loader = dataset_and_loader
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss_fn   = GroupContrastiveLoss(margin=1.0, mode="group_aware")

        for pyg_batch, labels, groups, shared, _ in loader:
            optimizer.zero_grad()
            out  = tiny_model(pyg_batch)
            loss = loss_fn(out, labels, groups, shared)
            loss.backward()
            optimizer.step()

        # If we reach here without an exception, the test passes.

    def test_model_parameters_change_after_step(self, tmp_path):
        """
        A gradient step must actually modify the model parameters.
        Uses a fully isolated fresh model + fresh dataset (via function-scoped
        tmp_path) so the test is independent of other tests' gradient steps.
        """
        # ── Build a fresh, isolated dataset ───────────────────────────────
        for cls in range(2):
            for grp in range(2):
                d = tmp_path / f"class_{cls}" / f"rep_{grp}"
                _write_pt_frames(str(d), prefix=f"iso_c{cls}g{grp}", n_frames=N_FRAMES)

        class_dirs = [str(tmp_path / "class_0"), str(tmp_path / "class_1")]
        ds = MDFlexibleWindowDataset(
            data_class_dirs=class_dirs,
            split="train",
            window_size=WINDOW_SIZE,
            window_offset=WINDOW_SIZE,
            ignore_validation_logic=True,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            collate_fn=collate_windows, num_workers=0)

        # ── Build a virgin model with a high LR to guarantee visible change ─
        fresh_model = HybridStSchnet(
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            window_size=WINDOW_SIZE,
            n_layers=N_LAYERS,
            cutoff=10.0,
        )
        fresh_model.train()

        # Snapshot initial parameter values
        before = {
            name: param.clone().detach()
            for name, param in fresh_model.named_parameters()
        }

        # Use 'base' mode so same-class pairs always attract regardless of
        # group structure, guaranteeing a nonzero loss and real gradients.
        optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
        loss_fn   = GroupContrastiveLoss(margin=1.0, mode="base")

        pyg_batch, labels, groups, shared, _ = next(iter(loader))
        optimizer.zero_grad()
        out  = fresh_model(pyg_batch)
        loss = loss_fn(out, labels, groups, shared)

        assert loss.item() > 0.0, (
            f"Loss is {loss.item():.6f} — no gradient will flow. "
            "Check that the batch contains both classes."
        )
        loss.backward()

        # Confirm gradients exist before stepping
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in fresh_model.parameters()
        ), "No nonzero gradients before optimizer.step()."

        optimizer.step()

        # At least one parameter must have changed
        changed = any(
            not torch.allclose(before[name], param.detach())
            for name, param in fresh_model.named_parameters()
        )
        assert changed, (
            "No model parameters changed after optimizer.step() — "
            "the update may have been silently skipped."
        )
