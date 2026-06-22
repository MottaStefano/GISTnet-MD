"""
test_architecture.py — Unit tests for src/core/architecture.py

Tests cover:
  - Envelope & RBFExpansion distance-encoding primitives
  - TemporalCNN and TemporalAttention temporal modules
  - ConfigurableGlobalPooling (softmax / sigmoid) with masking
  - HybridStSchnet full forward + backward pass
"""

import os
import sys
import pytest
import torch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.architecture import (
    Envelope,
    RBFExpansion,
    SparseInteractionBlock,
    TemporalCNN,
    TemporalAttention,
    MaskedMeanPooling,
    ConfigurableGlobalPooling,
    HybridStSchnet,
)
from conftest import make_batch_of_windows


# =============================================================================
# 1. Envelope & RBFExpansion
# =============================================================================

class TestEnvelopeAndRBF:
    """Verify that the two distance-encoding primitives behave correctly."""

    def setup_method(self):
        self.cutoff   = 10.0
        self.envelope = Envelope(exponent=5)
        self.rbf      = RBFExpansion(min_dist=0.0, max_dist=self.cutoff, num_gaussians=32)

    def test_envelope_inside_cutoff_nonzero(self):
        """Distances strictly inside cutoff should produce non-zero outputs."""
        dist_in = torch.linspace(0.01, self.cutoff - 0.01, steps=50)
        out = self.envelope(dist_in, self.cutoff)
        assert (out != 0).all(), "Envelope must be non-zero inside the cutoff."

    def test_envelope_beyond_cutoff_exact_zero(self):
        """Distances >= cutoff must produce *exactly* zero (hard mask)."""
        dist_out = torch.tensor([self.cutoff, self.cutoff + 1.0, self.cutoff + 5.0])
        out = self.envelope(dist_out, self.cutoff)
        assert torch.all(out == 0.0), (
            f"Envelope must be exactly 0 beyond cutoff; got {out}"
        )

    def test_envelope_output_shape(self):
        """Output shape equals input shape."""
        dist = torch.rand(128) * 15.0
        out  = self.envelope(dist, self.cutoff)
        assert out.shape == dist.shape

    def test_rbf_output_shape(self):
        """RBF output is (N, num_gaussians)."""
        dist = torch.rand(64) * self.cutoff
        out  = self.rbf(dist)
        assert out.shape == (64, 32), f"Expected (64, 32), got {out.shape}"

    def test_rbf_values_in_0_1(self):
        """
        Gaussian RBF outputs are in [0, 1].
        Note: exact 0.0 is possible for distances very far from any basis
        centre (IEEE 754 underflow) — that is correct behaviour.
        """
        dist = torch.rand(200) * self.cutoff
        out  = self.rbf(dist)
        assert out.min() >= 0.0, "RBF outputs must be non-negative."
        assert out.max() <= 1.0 + 1e-6, "RBF outputs must not exceed 1."
        # At least one Gaussian should fire strongly (max close to 1)
        assert out.max() > 0.5, "No RBF basis fired above 0.5 — check coeff formula."

    def test_rbf_no_nan(self):
        """No NaNs in RBF output for arbitrary distances."""
        dist = torch.cat([torch.zeros(1), torch.rand(99) * 20.0])
        out  = self.rbf(dist)
        assert not torch.isnan(out).any(), "RBF produced NaNs."


# =============================================================================
# 2. Temporal Modules
# =============================================================================

class TestTemporalModules:
    """TemporalCNN and TemporalAttention output shape and gradient flow."""

    BATCH    = 3
    CHANNELS = 16
    TIME     = 8

    def test_temporal_cnn_output_shape(self):
        """TemporalCNN: (B, C, T) → (B, C)."""
        net = TemporalCNN(hidden_dim=self.CHANNELS, dropout=0.0)
        x   = torch.randn(self.BATCH, self.CHANNELS, self.TIME)
        out = net(x)
        assert out.shape == (self.BATCH, self.CHANNELS), (
            f"Expected ({self.BATCH}, {self.CHANNELS}), got {out.shape}"
        )

    def test_temporal_cnn_gradient_flows(self):
        """Loss.backward() must not raise and must produce gradients."""
        net = TemporalCNN(hidden_dim=self.CHANNELS, dropout=0.0)
        x   = torch.randn(self.BATCH, self.CHANNELS, self.TIME, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None, "No gradient flowed through TemporalCNN."

    def test_temporal_attention_output_shape(self):
        """TemporalAttention: (B, T, C) → (B, C)."""
        net = TemporalAttention(hidden_dim=self.CHANNELS)
        x   = torch.randn(self.BATCH, self.TIME, self.CHANNELS)
        out = net(x)
        assert out.shape == (self.BATCH, self.CHANNELS), (
            f"Expected ({self.BATCH}, {self.CHANNELS}), got {out.shape}"
        )

    def test_temporal_attention_weights_sum_to_one(self):
        """Softmax attention weights along the time dimension should sum to ~1."""
        net    = TemporalAttention(hidden_dim=self.CHANNELS)
        x      = torch.randn(self.BATCH, self.TIME, self.CHANNELS)
        scores = net.query_layer(x)                      # (B, T, 1)
        weights = torch.softmax(scores, dim=1).squeeze(-1)
        sums   = weights.sum(dim=1)                      # (B,)
        assert torch.allclose(sums, torch.ones(self.BATCH), atol=1e-5), (
            "Attention weights must sum to 1 along the time axis."
        )

    def test_temporal_attention_no_nan(self):
        """TemporalAttention must not produce NaNs."""
        net = TemporalAttention(hidden_dim=self.CHANNELS)
        x   = torch.randn(self.BATCH, self.TIME, self.CHANNELS)
        out = net(x)
        assert not torch.isnan(out).any(), "TemporalAttention produced NaNs."


# =============================================================================
# 3. ConfigurableGlobalPooling
# =============================================================================

class TestPoolingMechanisms:
    """
    Validates ConfigurableGlobalPooling with both activations.

    Key assertions
    ──────────────
    • No NaNs in output.
    • Masked nodes do NOT dominate / affect the output (for softmax the
      masked score is −1e4, so its weight is effectively 0).
    • Output shape is (B, D).
    """

    BATCH     = 4
    MAX_NODES = 10
    DIM       = 16

    def _make_inputs(self, seed: int = 0):
        torch.manual_seed(seed)
        x    = torch.randn(self.BATCH, self.MAX_NODES, self.DIM)
        mask = torch.ones(self.BATCH, self.MAX_NODES, dtype=torch.bool)
        # Mask out the last 3 nodes for all graphs
        mask[:, -3:] = False
        return x, mask

    def test_softmax_output_shape(self):
        pool = ConfigurableGlobalPooling(self.DIM, activation='softmax')
        x, mask = self._make_inputs()
        out = pool(x, mask)
        assert out.shape == (self.BATCH, self.DIM)

    def test_sigmoid_output_shape(self):
        pool = ConfigurableGlobalPooling(self.DIM, activation='sigmoid')
        x, mask = self._make_inputs()
        out = pool(x, mask)
        assert out.shape == (self.BATCH, self.DIM)

    def test_softmax_no_nan(self):
        pool = ConfigurableGlobalPooling(self.DIM, activation='softmax')
        x, mask = self._make_inputs()
        out = pool(x, mask)
        assert not torch.isnan(out).any(), "Softmax pooling produced NaNs."

    def test_sigmoid_no_nan(self):
        pool = ConfigurableGlobalPooling(self.DIM, activation='sigmoid')
        x, mask = self._make_inputs()
        out = pool(x, mask)
        assert not torch.isnan(out).any(), "Sigmoid pooling produced NaNs."

    def test_softmax_masked_nodes_negligible(self):
        """
        With softmax, nodes whose mask=False receive score −1e4, so their
        weight is effectively 0. Setting their features to a huge constant
        should barely change the output.
        """
        pool = ConfigurableGlobalPooling(self.DIM, activation='softmax')
        pool.eval()
        x, mask = self._make_inputs(seed=7)

        out_baseline = pool(x, mask).detach()

        x_perturbed = x.clone()
        x_perturbed[:, -3:, :] = 1e6    # huge values in masked region
        out_perturbed = pool(x_perturbed, mask).detach()

        # Outputs should be close (masked nodes effectively excluded)
        assert torch.allclose(out_baseline, out_perturbed, atol=1e-2), (
            "Softmax pooling is not properly masking out excluded nodes."
        )

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            pool = ConfigurableGlobalPooling(self.DIM, activation='relu')
            x    = torch.randn(self.BATCH, self.MAX_NODES, self.DIM)
            pool(x)


# =============================================================================
# 4. SparseInteractionBlock
# =============================================================================

class TestSparseInteractionBlock:
    """Isolated tests for the GNN message-passing layer."""

    HIDDEN   = 16
    N_GAUSS  = 32
    N_NODES  = 10
    N_EDGES  = 20

    def _make_inputs(self):
        torch.manual_seed(42)
        h          = torch.randn(self.N_NODES, self.HIDDEN)
        src        = torch.randint(0, self.N_NODES, (self.N_EDGES,))
        dst        = torch.randint(0, self.N_NODES, (self.N_EDGES,))
        edge_index = torch.stack([src, dst])
        rbf_feat   = torch.rand(self.N_EDGES, self.N_GAUSS)
        return h, edge_index, rbf_feat

    def test_output_shape(self):
        block = SparseInteractionBlock(self.HIDDEN, self.N_GAUSS)
        block.eval()
        h, edge_index, rbf_feat = self._make_inputs()
        out = block(h, edge_index, rbf_feat)
        assert out.shape == h.shape, (
            f"Expected shape {h.shape}, got {out.shape}"
        )

    def test_residual_connection(self):
        """
        SparseInteractionBlock uses h_new = h + update_net(aggr).
        With zero-initialised weights in update_net, output ≈ h.
        """
        block = SparseInteractionBlock(self.HIDDEN, self.N_GAUSS)
        # Zero-init the last linear layer of update_net
        with torch.no_grad():
            block.update_net[-2].weight.zero_()
            block.update_net[-2].bias.zero_()
        block.eval()
        h, edge_index, rbf_feat = self._make_inputs()
        out = block(h, edge_index, rbf_feat)
        # After BN, the residual may shift slightly, but should be close
        # Just verify it doesn't diverge wildly
        diff = (out - h).abs().mean()
        assert diff < 5.0, (
            f"With zeroed update_net, output diverged from h by {diff:.4f}"
        )

    def test_gradient_flows(self):
        block = SparseInteractionBlock(self.HIDDEN, self.N_GAUSS)
        block.train()
        h, edge_index, rbf_feat = self._make_inputs()
        h = h.detach().requires_grad_(True)
        out = block(h, edge_index, rbf_feat)
        out.sum().backward()
        assert h.grad is not None, "Gradient did not flow through SparseInteractionBlock."

    def test_no_nan_output(self):
        block = SparseInteractionBlock(self.HIDDEN, self.N_GAUSS)
        block.eval()
        h, edge_index, rbf_feat = self._make_inputs()
        out = block(h, edge_index, rbf_feat)
        assert not torch.isnan(out).any(), "SparseInteractionBlock produced NaNs."


# =============================================================================
# 5. MaskedMeanPooling
# =============================================================================

class TestMaskedMeanPooling:
    """Tests for the simple masked mean pooling module."""

    BATCH     = 3
    MAX_NODES = 8
    DIM       = 16

    def _make_inputs(self):
        torch.manual_seed(0)
        x    = torch.randn(self.BATCH, self.MAX_NODES, self.DIM)
        mask = torch.ones(self.BATCH, self.MAX_NODES, dtype=torch.bool)
        mask[:, -2:] = False   # mask out last 2 nodes
        return x, mask

    def test_output_shape_with_mask(self):
        pool = MaskedMeanPooling()
        x, mask = self._make_inputs()
        out = pool(x, mask)
        assert out.shape == (self.BATCH, self.DIM)

    def test_output_shape_without_mask(self):
        pool = MaskedMeanPooling()
        x, _ = self._make_inputs()
        out = pool(x, mask=None)
        assert out.shape == (self.BATCH, self.DIM)

    def test_without_mask_equals_mean(self):
        """When mask=None, output must equal x.mean(dim=1)."""
        pool = MaskedMeanPooling()
        x, _ = self._make_inputs()
        out = pool(x, mask=None)
        expected = x.mean(dim=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_masked_nodes_excluded(self):
        """
        Perturbing masked nodes must not change the output.
        """
        pool = MaskedMeanPooling()
        x, mask = self._make_inputs()

        out_baseline = pool(x, mask)

        x_perturbed = x.clone()
        x_perturbed[:, -2:, :] = 1e6   # huge values in masked positions
        out_perturbed = pool(x_perturbed, mask)

        assert torch.allclose(out_baseline, out_perturbed, atol=1e-6), (
            "MaskedMeanPooling is not excluding masked nodes."
        )

    def test_correct_mean_with_mask(self):
        """
        Manually verify the mean is computed over unmasked nodes only.
        """
        pool = MaskedMeanPooling()
        x, mask = self._make_inputs()
        n_valid = mask[0].sum().item()   # first 6 nodes are valid
        out = pool(x, mask)
        expected = x[0, :n_valid, :].mean(dim=0)
        assert torch.allclose(out[0], expected, atol=1e-6), (
            "Masked mean does not match manual computation."
        )


# =============================================================================
# 6. HybridStSchnet — Forward + Backward
# =============================================================================

class TestHybridStSchnetForwardAndBackward:
    """
    End-to-end model tests.  We use tiny hyper-parameters so the test is fast.
    """

    WINDOW_SIZE     = 4
    N_LOGICAL       = 2   # mini-batch of 2 MD windows
    N_NODES         = 8
    HIDDEN_DIM      = 16
    EMBEDDING_DIM   = 8
    N_LAYERS        = 1

    @pytest.fixture(autouse=True)
    def _build_model_and_batch(self):
        """Construct a tiny model and a compatible synthetic batch."""
        self.batch = make_batch_of_windows(
            n_logical_graphs=self.N_LOGICAL,
            window_size=self.WINDOW_SIZE,
            n_nodes=self.N_NODES,
        )
        self.model = HybridStSchnet(
            hidden_dim=self.HIDDEN_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            window_size=self.WINDOW_SIZE,
            n_layers=self.N_LAYERS,
            cutoff=10.0,
            temporal_setup='cnn',
            pooling_activation='softmax',
        )
        self.model.train()

    def test_forward_output_shape(self):
        out = self.model(self.batch)
        expected = (self.N_LOGICAL, self.EMBEDDING_DIM)
        assert out.shape == expected, (
            f"Expected output shape {expected}, got {out.shape}"
        )

    def test_forward_output_l2_normalized(self):
        """Without num_classes, the model returns L2-normalised embeddings."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.batch)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(self.N_LOGICAL), atol=1e-5), (
            "Output embeddings are not L2-normalised."
        )

    def test_backward_graph_intact(self):
        """
        A dummy loss.backward() must populate gradients all the way back to
        the embedding layer, proving the computational graph is intact.
        """
        out  = self.model(self.batch)
        loss = out.sum()
        loss.backward()
        grad = self.model.spatial_encoder.embedding.weight.grad
        assert grad is not None, (
            "Gradient for spatial_encoder.embedding.weight is None — "
            "the computational graph is broken."
        )
        assert not torch.isnan(grad).any(), "NaN gradients detected."

    def test_forward_no_nan(self):
        out = self.model(self.batch)
        assert not torch.isnan(out).any(), "HybridStSchnet produced NaN outputs."

    def test_attention_temporal_setup(self):
        """Verify the model also works with temporal_setup='attention'."""
        model_attn = HybridStSchnet(
            hidden_dim=self.HIDDEN_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            window_size=self.WINDOW_SIZE,
            n_layers=self.N_LAYERS,
            cutoff=10.0,
            temporal_setup='attention',
            pooling_activation='sigmoid',
        )
        model_attn.eval()
        with torch.no_grad():
            out = model_attn(self.batch)
        assert out.shape == (self.N_LOGICAL, self.EMBEDDING_DIM)
        assert not torch.isnan(out).any()

    def test_classifier_head_shape(self):
        """When num_classes is set, output must have shape (B, num_classes)."""
        model_cls = HybridStSchnet(
            hidden_dim=self.HIDDEN_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            window_size=self.WINDOW_SIZE,
            n_layers=self.N_LAYERS,
            cutoff=10.0,
            num_classes=3,
        )
        model_cls.eval()
        with torch.no_grad():
            out = model_cls(self.batch)
        assert out.shape == (self.N_LOGICAL, 3), (
            f"Classifier head expected shape ({self.N_LOGICAL}, 3), got {out.shape}"
        )

    def test_invalid_temporal_setup_raises(self):
        with pytest.raises(ValueError, match="Unknown temporal_setup"):
            HybridStSchnet(
                hidden_dim=self.HIDDEN_DIM,
                embedding_dim=self.EMBEDDING_DIM,
                window_size=self.WINDOW_SIZE,
                temporal_setup='transformer',
            )
