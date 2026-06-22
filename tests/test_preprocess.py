"""
test_preprocess.py — Unit tests for src/preprocess.py

Tests cover:
  - parse_selection_string: correct parsing, empty input, malformed strings
  - process_frame_task: saves a valid .pt file without touching mdtraj
"""

import os
import sys
import pytest
import torch
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import only the public helpers — avoids triggering mdtraj's heavy __main__
# block (preprocess.py guards heavy imports behind `if __name__ == "__main__"`)
from preprocess import parse_selection_string, process_frame_task


# =============================================================================
# 1. parse_selection_string
# =============================================================================

class TestParseSelectionString:
    """Unit tests for the parse_selection_string utility."""

    def test_simple_range(self):
        result = parse_selection_string("1-3")
        assert result == [1, 2, 3], f"Got {result}"

    def test_single_residue(self):
        result = parse_selection_string("5")
        assert result == [5]

    def test_range_and_single(self):
        result = parse_selection_string("1-3, 5")
        assert result == [1, 2, 3, 5]

    def test_multiple_ranges(self):
        result = parse_selection_string("1-3, 7-9")
        assert result == [1, 2, 3, 7, 8, 9]

    def test_result_is_sorted(self):
        result = parse_selection_string("10, 1-3")
        assert result == sorted(result), "Output must be sorted."

    def test_duplicates_removed(self):
        """Overlapping ranges should not produce duplicates."""
        result = parse_selection_string("1-3, 2-4")
        assert result == sorted(set(result)), "Duplicates must be removed."

    def test_whitespace_around_commas_tolerated(self):
        """Spaces around commas are stripped; result is correct."""
        result = parse_selection_string("  1-3 ,  5  ")
        assert result == [1, 2, 3, 5]

    def test_spaces_around_dash_tolerated(self):
        """
        Spaces around '-' produce tokens like '1 ' and ' 3'.
        Since int() strips whitespace natively, this is gracefully
        handled without raising an error.
        """
        result = parse_selection_string("1 - 3")
        assert result == [1, 2, 3]

    def test_empty_string_returns_empty_list(self):
        result = parse_selection_string("")
        assert result == [], f"Empty string must return [], got {result}."

    def test_only_commas_returns_empty_list(self):
        result = parse_selection_string(",,,")
        assert result == []

    def test_malformed_range_raises_value_error(self):
        """'1-abc' is not a valid range and must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid range format"):
            parse_selection_string("1-abc")

    def test_non_numeric_raises_value_error(self):
        """A plain non-numeric token must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid residue number"):
            parse_selection_string("XYZ")

    def test_large_range(self):
        result = parse_selection_string("1-100")
        assert len(result) == 100
        assert result[0] == 1 and result[-1] == 100


# =============================================================================
# 2. process_frame_task
# =============================================================================

class TestProcessFrameTask:
    """
    Call process_frame_task directly with synthetic numpy arrays.
    No mdtraj, no .xtc / .pdb files involved.
    """

    # --- Shared parameters ---
    N_NODES     = 10
    CUTOFF      = 10.0
    MAX_K       = 32
    REPR        = "ca"
    NODE_DIHE   = "none"
    FRAME_IDX   = 42
    PREFIX      = "test_sim"

    def _make_task_args(self, out_dir: str, n_nodes: int = None):
        """Build the args tuple expected by process_frame_task."""
        n = n_nodes or self.N_NODES
        np.random.seed(0)
        # Coordinates in nm (preprocess_frame_task converts to Å internally: * 10)
        coords_nm      = np.random.rand(n, 3).astype(np.float32)
        static_indices = torch.randint(0, 20, (n,))
        dihe_feats     = None   # node_dihe='none'
        return (
            coords_nm,
            static_indices,
            dihe_feats,
            self.FRAME_IDX,
            out_dir,
            self.PREFIX,
            self.CUTOFF,
            self.MAX_K,
            self.REPR,
            self.NODE_DIHE,
        )

    def test_output_file_created(self, tmp_path):
        """process_frame_task must save a .pt file in out_dir."""
        args   = self._make_task_args(str(tmp_path))
        result = process_frame_task(args)

        expected_name = f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        expected_path = tmp_path / expected_name

        assert result is None, (
            f"process_frame_task returned an error: {result}"
        )
        assert expected_path.exists(), (
            f"Expected output file {expected_path} was not created."
        )

    def test_saved_file_is_pyg_data(self, tmp_path):
        """Loading the saved .pt file must yield a PyG Data object."""
        from torch_geometric.data import Data

        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)
        assert isinstance(data, Data), (
            f"Loaded object is not a PyG Data, got {type(data)}."
        )

    def test_data_has_edge_index(self, tmp_path):
        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)
        assert hasattr(data, "edge_index") and data.edge_index is not None
        assert data.edge_index.shape[0] == 2, "edge_index must be 2 × E."

    def test_data_has_correct_node_count(self, tmp_path):
        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)
        assert data.num_nodes == self.N_NODES, (
            f"Expected {self.N_NODES} nodes, got {data.num_nodes}."
        )

    def test_metadata_embedded(self, tmp_path):
        """The saved graph must carry cutoff, representation, node_dihe metadata."""
        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)

        assert hasattr(data, "cutoff"),         "cutoff metadata missing."
        assert hasattr(data, "representation"), "representation metadata missing."
        assert hasattr(data, "node_dihe"),      "node_dihe metadata missing."

        assert data.cutoff == self.CUTOFF
        assert data.representation == self.REPR
        assert data.node_dihe == self.NODE_DIHE

    def test_edges_within_cutoff(self, tmp_path):
        """All saved edges must have distance ≤ cutoff (+ small tolerance)."""
        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)

        assert data.edge_attr.max().item() <= self.CUTOFF + 1e-4, (
            "Some edges exceed the cutoff distance."
        )

    def test_coordinates_converted_to_angstroms(self, tmp_path):
        """
        Positions in the saved Data object should be in Ångströms.
        Input coords_nm are in [0, 1) nm → pos should be in [0, 10) Å.
        """
        args = self._make_task_args(str(tmp_path))
        process_frame_task(args)

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)

        assert data.pos.max().item() < 10.0 + 1e-3, (
            "Positions appear to still be in nm rather than Å."
        )

    def test_returns_none_on_success(self, tmp_path):
        """process_frame_task must return None (not an error string) on success."""
        args   = self._make_task_args(str(tmp_path))
        result = process_frame_task(args)
        assert result is None

    def test_dihedral_features_saved_when_provided(self, tmp_path):
        """
        If dihe_feats is not None, the saved Data object must have x_dihe.
        """
        n = self.N_NODES
        np.random.seed(1)
        coords_nm      = np.random.rand(n, 3).astype(np.float32)
        static_indices = torch.randint(0, 20, (n,))
        dihe_feats     = np.random.rand(n, 4).astype(np.float32)  # 4 dihedral features

        args = (
            coords_nm,
            static_indices,
            dihe_feats,
            self.FRAME_IDX,
            str(tmp_path),
            self.PREFIX,
            self.CUTOFF,
            self.MAX_K,
            self.REPR,
            "backbone",
        )
        result = process_frame_task(args)
        assert result is None

        fpath = tmp_path / f"{self.PREFIX}_frame_{self.FRAME_IDX:06d}.pt"
        data  = torch.load(str(fpath), weights_only=False)
        assert hasattr(data, "x_dihe"), "x_dihe attribute missing when dihedrals provided."
        assert data.x_dihe.shape == (n, 4)
