"""
test_dataset.py — Unit tests for src/core/dataset.py

Tests cover:
  - MDFlexibleWindowDataset discovery and windowing logic
  - Window skip / stride / val_groups parameters
  - get_weights class-balanced sampling
  - collate_windows collation function
"""

import os
import sys
import pytest
import torch
from torch_geometric.data import Data, Batch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.dataset import MDFlexibleWindowDataset, collate_windows
from conftest import make_single_graph, write_pt_frames


# =============================================================================
# 1. Dataset Discovery
# =============================================================================

class TestDatasetDiscovery:
    """
    Verify that MDFlexibleWindowDataset correctly
     (a) discovers .pt files under nested directories, and
     (b) computes the number of windows from window_size and window_offset.
    """

    def test_single_class_window_count(self, tmp_path):
        """
        With N frames, window_size W, and stride S (= window_offset):
          expected_windows = (N - W) // S + 1   when (N - W) % S == 0
          = floor((N - W) / S) + 1              in general
        """
        n_frames   = 20
        window_size = 5
        stride      = 3          # window_offset defaults to window_size if not set

        class_dir  = tmp_path / "class_0" / "group_A"
        write_pt_frames(str(class_dir), prefix="traj", n_frames=n_frames, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train',
            window_size=window_size,
            window_offset=stride,   # explicit step between windows
            stride=1,               # frame sub-sampling (keep all)
            ignore_validation_logic=True,
        )

        # Expected: positions 0, 3, 6, 9, 12 → 5 windows (5+3=8,11,14 each ≤ 20-5=15)
        expected = (n_frames - window_size) // stride + 1
        assert len(ds) == expected, (
            f"Expected {expected} windows, got {len(ds)}"
        )

    def test_two_classes_combined_count(self, tmp_path):
        """Two class directories should produce independent window sequences."""
        n_frames    = 12
        window_size = 4
        stride      = 2

        for cls in range(2):
            class_dir = tmp_path / f"class_{cls}" / "rep_1"
            write_pt_frames(str(class_dir), prefix="sim", n_frames=n_frames, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[
                str(tmp_path / "class_0"),
                str(tmp_path / "class_1"),
            ],
            split='train',
            window_size=window_size,
            window_offset=stride,
            ignore_validation_logic=True,
        )

        windows_per_class = (n_frames - window_size) // stride + 1
        assert len(ds) == 2 * windows_per_class, (
            f"Expected {2 * windows_per_class} total windows, got {len(ds)}"
        )

    def test_stride_subsamples_frames(self, tmp_path):
        """
        stride=2 should halve the effective number of frames before windowing,
        yielding fewer windows than stride=1.
        """
        n_frames   = 20
        window_size = 4

        class_dir = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(class_dir), prefix="traj", n_frames=n_frames, n_nodes=6)

        root = str(tmp_path / "class_0")

        ds_stride1 = MDFlexibleWindowDataset(
            data_class_dirs=[root], split='train',
            window_size=window_size, window_offset=1, stride=1,
            ignore_validation_logic=True,
        )
        ds_stride2 = MDFlexibleWindowDataset(
            data_class_dirs=[root], split='train',
            window_size=window_size, window_offset=1, stride=2,
            ignore_validation_logic=True,
        )
        assert len(ds_stride2) < len(ds_stride1), (
            "stride=2 must produce fewer windows than stride=1."
        )

    def test_getitem_returns_expected_structure(self, tmp_path):
        """
        __getitem__ must return a 5-tuple:
        (graph_list, label, group_id, shared_name_id, paths)
        """
        n_frames   = 6
        window_size = 3

        class_dir = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(class_dir), prefix="sim", n_frames=n_frames, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train',
            window_size=window_size,
            window_offset=1,
            ignore_validation_logic=True,
        )

        item = ds[0]
        assert len(item) == 5, "Expected 5-tuple from __getitem__."
        graph_list, label, group_id, shared_name_id, paths = item
        assert len(graph_list) == window_size, (
            f"Window must contain {window_size} graphs, got {len(graph_list)}."
        )
        assert isinstance(label, int)
        assert len(paths) == window_size

    def test_missing_directory_skipped_gracefully(self, tmp_path):
        """A non-existent data directory should not crash the dataset."""
        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "nonexistent")],
            split='train',
            window_size=4,
            ignore_validation_logic=True,
        )
        assert len(ds) == 0, "Dataset from missing dir must be empty."

    def test_files_without_frame_pattern_ignored(self, tmp_path):
        """Files that don't match the expected naming convention are skipped."""
        class_dir = tmp_path / "class_0" / "rep_1"
        os.makedirs(str(class_dir))
        # Write some files with wrong names
        for name in ["bad_name.pt", "another.pt", "readme.txt"]:
            torch.save({}, os.path.join(str(class_dir), name))

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train',
            window_size=1,
            ignore_validation_logic=True,
        )
        assert len(ds) == 0, (
            "Files not matching the frame-naming convention must be ignored."
        )


# =============================================================================
# 2. skip parameter
# =============================================================================

class TestSkipParameter:
    """Verify that the `skip` parameter discards early frames."""

    def test_skip_drops_early_frames(self, tmp_path):
        """
        With 20 frames and skip=5, only frames 5..19 are available (15 frames).
        Windows are then created from this reduced set.
        """
        n_frames    = 20
        skip        = 5
        window_size = 4

        class_dir = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(class_dir), prefix="sim", n_frames=n_frames, n_nodes=6)

        ds_no_skip = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train', window_size=window_size, window_offset=window_size,
            ignore_validation_logic=True, skip=0,
        )
        ds_with_skip = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train', window_size=window_size, window_offset=window_size,
            ignore_validation_logic=True, skip=skip,
        )

        assert len(ds_with_skip) < len(ds_no_skip), (
            f"skip={skip} must produce fewer windows. "
            f"Got no_skip={len(ds_no_skip)}, skip={len(ds_with_skip)}"
        )

    def test_skip_all_frames_yields_empty(self, tmp_path):
        """If skip >= n_frames, the dataset must be empty."""
        n_frames = 10
        class_dir = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(class_dir), prefix="sim", n_frames=n_frames, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train', window_size=4, ignore_validation_logic=True,
            skip=n_frames,  # skip all
        )
        assert len(ds) == 0, "Skipping all frames must yield an empty dataset."


# =============================================================================
# 3. Train / Validation split via val_groups
# =============================================================================

class TestValGroupsSplit:
    """Verify that val_groups correctly partitions data into train and val."""

    def test_val_split_excludes_groups(self, tmp_path):
        """
        Two groups under the same class. With val_groups=[1], group 1
        (first sorted group) goes to val; group 2 goes to train.
        """
        n_frames    = 12
        window_size = 4

        for grp_name in ["rep_A", "rep_B"]:
            d = tmp_path / "class_0" / grp_name
            write_pt_frames(str(d), prefix="sim", n_frames=n_frames, n_nodes=6)

        root = [str(tmp_path / "class_0")]

        ds_train = MDFlexibleWindowDataset(
            data_class_dirs=root, split='train',
            window_size=window_size, window_offset=window_size,
            val_groups=[1],   # group 1 (first sorted) → validation
        )
        ds_val = MDFlexibleWindowDataset(
            data_class_dirs=root, split='val',
            window_size=window_size, window_offset=window_size,
            val_groups=[1],
        )

        assert len(ds_train) > 0, "Train split must not be empty."
        assert len(ds_val) > 0,   "Val split must not be empty."
        assert len(ds_train) + len(ds_val) > 0

    def test_no_val_groups_puts_everything_in_train(self, tmp_path):
        """Without val_groups, all data should go to train."""
        n_frames    = 12
        window_size = 4

        d = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(d), prefix="sim", n_frames=n_frames, n_nodes=6)

        root = [str(tmp_path / "class_0")]

        ds_train = MDFlexibleWindowDataset(
            data_class_dirs=root, split='train',
            window_size=window_size, window_offset=window_size,
        )
        ds_val = MDFlexibleWindowDataset(
            data_class_dirs=root, split='val',
            window_size=window_size, window_offset=window_size,
        )
        assert len(ds_train) > 0
        assert len(ds_val) == 0, "Val split must be empty with no val_groups."


# =============================================================================
# 4. get_weights
# =============================================================================

class TestGetWeights:
    """Verify class-balanced sampling weights from get_weights()."""

    def test_get_weights_inverse_frequency(self, tmp_path):
        """
        With 2 classes and different window counts, weights must be
        inversely proportional to class frequency.
        """
        window_size = 4

        # Class 0: 20 frames → 5 non-overlapping windows
        d0 = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(d0), prefix="sim", n_frames=20, n_nodes=6)

        # Class 1: 8 frames → 2 non-overlapping windows
        d1 = tmp_path / "class_1" / "rep_1"
        write_pt_frames(str(d1), prefix="sim", n_frames=8, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0"), str(tmp_path / "class_1")],
            split='train', window_size=window_size, window_offset=window_size,
            ignore_validation_logic=True,
        )

        weights = ds.get_weights()
        assert weights.dim() == 1, "Weights must be a 1-D tensor."
        assert weights.shape[0] == len(ds), "One weight per sample."
        assert (weights > 0).all(), "All weights must be positive."

        # The minority class (class 1) should have higher per-sample weight
        n_class0 = ds.class_counts[0]
        n_class1 = ds.class_counts[1]
        w_class0 = weights[0].item()   # weight for a class-0 sample
        w_class1 = weights[n_class0].item()  # weight for a class-1 sample

        if n_class0 > n_class1:
            assert w_class1 > w_class0, (
                "Minority class must have higher weight than majority class."
            )

    def test_get_weights_single_class(self, tmp_path):
        """With one class only, all weights should be equal."""
        d = tmp_path / "class_0" / "rep_1"
        write_pt_frames(str(d), prefix="sim", n_frames=12, n_nodes=6)

        ds = MDFlexibleWindowDataset(
            data_class_dirs=[str(tmp_path / "class_0")],
            split='train', window_size=4, window_offset=4,
            ignore_validation_logic=True,
        )
        weights = ds.get_weights()
        assert torch.allclose(weights, weights[0].expand_as(weights)), (
            "All weights must be equal for a single-class dataset."
        )


# =============================================================================
# 5. collate_windows
# =============================================================================

class TestCollateWindows:
    """
    Verify that collate_windows correctly combines a list of dataset items into
    a batched representation.
    """

    def _make_item(self, label: int, group_id: int, shared_id: int,
                   window_size: int = 3, n_nodes: int = 6):
        """Build a synthetic dataset __getitem__ return value."""
        graph_list = [make_single_graph(n_nodes=n_nodes, seed=label * 100 + t)
                      for t in range(window_size)]
        # Fake paths
        paths = [f"/fake/path/class_{label}_frame_{t:06d}.pt" for t in range(window_size)]
        return graph_list, label, group_id, shared_id, paths

    def test_collate_returns_valid_batch(self):
        """collate_windows must return a PyG Batch object."""
        items = [self._make_item(0, 0, 0), self._make_item(1, 1, 1)]
        batch, labels, groups, shared, paths = collate_windows(items)
        assert isinstance(batch, Batch), "First return value must be a PyG Batch."

    def test_collate_labels_are_1d_tensor(self):
        items = [self._make_item(0, 0, 0), self._make_item(1, 1, 1)]
        _, labels, _, _, _ = collate_windows(items)
        assert labels.dim() == 1, f"Labels must be 1-D, got {labels.dim()}-D."
        assert labels.shape == (2,), f"Expected shape (2,), got {labels.shape}."

    def test_collate_groups_are_1d_tensor(self):
        items = [self._make_item(0, 3, 0), self._make_item(1, 5, 1)]
        _, _, groups, _, _ = collate_windows(items)
        assert groups.dim() == 1
        assert groups.tolist() == [3, 5], (
            f"Groups mismatch: expected [3, 5], got {groups.tolist()}"
        )

    def test_collate_batch_node_count(self):
        """
        Batch should contain (n_items × window_size × n_nodes) nodes total.
        """
        n_items     = 3
        window_size = 4
        n_nodes     = 6
        items = [
            self._make_item(i, i, i, window_size=window_size, n_nodes=n_nodes)
            for i in range(n_items)
        ]
        batch, _, _, _, _ = collate_windows(items)
        expected_nodes = n_items * window_size * n_nodes
        assert batch.num_nodes == expected_nodes, (
            f"Expected {expected_nodes} total nodes in batch, got {batch.num_nodes}."
        )

    def test_collate_preserves_label_values(self):
        """Labels must exactly match the values supplied to each item."""
        items = [
            self._make_item(0, 0, 0),
            self._make_item(0, 1, 0),
            self._make_item(1, 2, 1),
        ]
        _, labels, _, _, _ = collate_windows(items)
        assert labels.tolist() == [0, 0, 1]

    def test_collate_batch_has_edge_index(self):
        """Resulting batch must carry a valid edge_index attribute."""
        items = [self._make_item(0, 0, 0)]
        batch, _, _, _, _ = collate_windows(items)
        assert hasattr(batch, "edge_index"), "Batch is missing edge_index."
        assert batch.edge_index.shape[0] == 2

    def test_collate_shared_names_tensor(self):
        """shared_names must be a 1-D tensor with correct values."""
        items = [self._make_item(0, 0, 7), self._make_item(1, 1, 9)]
        _, _, _, shared, _ = collate_windows(items)
        assert shared.dim() == 1
        assert shared.tolist() == [7, 9]
