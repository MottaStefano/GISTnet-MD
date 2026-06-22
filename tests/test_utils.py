"""
test_utils.py — Unit tests for src/core/utils.py

Tests cover:
  - parse_with_config: config-file values and CLI override behaviour
  - AverageMeter: accumulator arithmetic
"""

import os
import sys
import pytest
import argparse
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.utils import (
    parse_with_config,
    save_and_reload_config,
    seed_everything,
    parse_mask_string,
    AverageMeter,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_parser() -> argparse.ArgumentParser:
    """
    Minimal parser that mirrors a subset of the preprocess.py CLI.
    Used to test parse_with_config in isolation.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None)
    p.add_argument("--cutoff",       type=float, default=10.0)
    p.add_argument("--representation", choices=["ca", "cb", "com"], default="ca")
    p.add_argument("--n_layers",     type=int,   default=3)
    p.add_argument("--flag",         action="store_true", default=False)
    return p


# =============================================================================
# 1. parse_with_config
# =============================================================================

class TestConfigParsing:
    """Verify that parse_with_config blends config-file values with CLI overrides."""

    def test_defaults_when_no_config(self, tmp_path):
        """With no --config, argparse defaults must be used."""
        with patch("sys.argv", ["prog"]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.cutoff == 10.0
        assert args.representation == "ca"
        assert args.n_layers == 3

    def test_config_file_overrides_defaults(self, tmp_path):
        """Values in the config file must override argparse defaults."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text(
            "cutoff=15.0\n"
            "representation=cb\n"
            "n_layers=5\n"
        )
        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)

        assert args.cutoff         == 15.0, f"cutoff: expected 15.0, got {args.cutoff}"
        assert args.representation == "cb",  f"representation: expected 'cb', got {args.representation}"
        assert args.n_layers       == 5,     f"n_layers: expected 5, got {args.n_layers}"

    def test_cli_overrides_config_file(self, tmp_path):
        """CLI arguments must take priority over config-file values."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text("cutoff=15.0\nrepresentation=cb\n")

        with patch("sys.argv", ["prog", "--config", str(cfg), "--cutoff", "20.0"]):
            parser = _make_parser()
            args   = parse_with_config(parser)

        assert args.cutoff == 20.0, (
            f"CLI --cutoff=20.0 must override config cutoff=15.0, got {args.cutoff}"
        )
        # representation should still come from the config (not overridden on CLI)
        assert args.representation == "cb"

    def test_config_comment_lines_ignored(self, tmp_path):
        """Lines starting with '#' are comments and must be ignored."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text(
            "# This is a comment\n"
            "cutoff=8.0\n"
            "# another comment\n"
        )
        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.cutoff == 8.0

    def test_config_inline_comments_stripped(self, tmp_path):
        """Inline comments (after ` #`) should be stripped from values."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text("cutoff=12.5 # a trailing comment\n")

        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.cutoff == 12.5, (
            f"Inline comment not stripped; cutoff={args.cutoff}"
        )

    def test_boolean_true_parsed(self, tmp_path):
        """'true' in the config file should set a bool arg to True."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text("flag=true\n")

        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.flag is True

    def test_boolean_false_parsed(self, tmp_path):
        """'false' in the config file should set a bool arg to False."""
        cfg = tmp_path / "test.cfg"
        cfg.write_text("flag=false\n")

        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.flag is False

    def test_missing_config_file_prints_warning(self, tmp_path, capsys):
        """A non-existent config path should print a warning but not crash."""
        missing = str(tmp_path / "does_not_exist.cfg")
        with patch("sys.argv", ["prog", "--config", missing]):
            parser = _make_parser()
            args   = parse_with_config(parser)

        captured = capsys.readouterr()
        assert "Warning" in captured.out, (
            "Expected a warning message for missing config file."
        )
        # Defaults should still be in effect
        assert args.cutoff == 10.0

    def test_empty_config_file_uses_defaults(self, tmp_path):
        """An empty config file must not alter any defaults."""
        cfg = tmp_path / "empty.cfg"
        cfg.write_text("")

        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)
        assert args.cutoff == 10.0
        assert args.n_layers == 3

    def test_partial_config_does_not_override_unmentioned_defaults(self, tmp_path):
        """
        Only the keys listed in the config file should be updated;
        unmentioned args must keep their argparse defaults.
        """
        cfg = tmp_path / "partial.cfg"
        cfg.write_text("cutoff=7.5\n")

        with patch("sys.argv", ["prog", "--config", str(cfg)]):
            parser = _make_parser()
            args   = parse_with_config(parser)

        assert args.cutoff     == 7.5    # updated by config
        assert args.n_layers   == 3      # default preserved
        assert args.representation == "ca"  # default preserved


# =============================================================================
# 2. AverageMeter
# =============================================================================

class TestAverageMeter:
    """Verify that AverageMeter correctly tracks running averages."""

    def test_initial_state(self):
        meter = AverageMeter("loss")
        assert meter.val == 0 and meter.avg == 0 and meter.count == 0

    def test_single_update(self):
        meter = AverageMeter("loss")
        meter.update(4.0)
        assert meter.val == 4.0
        assert meter.avg == 4.0
        assert meter.count == 1

    def test_multiple_updates_average(self):
        meter = AverageMeter("loss")
        for v in [2.0, 4.0, 6.0]:
            meter.update(v)
        assert meter.avg == pytest.approx(4.0)

    def test_weighted_update(self):
        """update(val, n) weights the sample by n."""
        meter = AverageMeter("loss")
        meter.update(10.0, n=5)  # contributes 50 to sum
        meter.update(0.0,  n=5)  # contributes 0
        assert meter.avg == pytest.approx(5.0)

    def test_reset_clears_state(self):
        meter = AverageMeter("loss")
        meter.update(99.0)
        meter.reset()
        assert meter.val == 0 and meter.avg == 0 and meter.count == 0

    def test_str_representation(self):
        """__str__ should not raise."""
        meter = AverageMeter("loss", fmt=":.4f")
        meter.update(3.14)
        assert "loss" in str(meter)


# =============================================================================
# 3. seed_everything
# =============================================================================

class TestSeedEverything:
    """Verify that seed_everything produces deterministic outputs."""

    def test_torch_determinism(self):
        """Two calls with the same seed must produce identical random tensors."""
        import torch
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert torch.allclose(a, b), (
            "torch.randn not deterministic after seed_everything."
        )

    def test_numpy_determinism(self):
        """NumPy random must also be seeded."""
        import numpy as np
        seed_everything(99)
        a = np.random.rand(10)
        seed_everything(99)
        b = np.random.rand(10)
        assert (a == b).all(), "numpy.random not deterministic after seed_everything."

    def test_python_random_determinism(self):
        """Python's built-in random must also be seeded."""
        import random
        seed_everything(77)
        a = [random.random() for _ in range(10)]
        seed_everything(77)
        b = [random.random() for _ in range(10)]
        assert a == b, "random.random not deterministic after seed_everything."

    def test_different_seeds_differ(self):
        """Different seeds must produce different outputs."""
        import torch
        seed_everything(1)
        a = torch.randn(5)
        seed_everything(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b), (
            "Different seeds produced identical tensors."
        )


# =============================================================================
# 4. parse_mask_string
# =============================================================================

class TestParseMaskString:
    """
    parse_mask_string is similar to parse_selection_string but with
    *silent* error handling (no ValueError raised).
    """

    def test_simple_range(self):
        assert parse_mask_string("1-3") == [1, 2, 3]

    def test_single_value(self):
        assert parse_mask_string("5") == [5]

    def test_mixed(self):
        result = parse_mask_string("1-3, 7, 10-12")
        assert result == [1, 2, 3, 7, 10, 11, 12]

    def test_empty_string_returns_empty(self):
        assert parse_mask_string("") == []

    def test_none_returns_empty(self):
        assert parse_mask_string(None) == []

    def test_malformed_range_silently_skipped(self):
        """
        Unlike parse_selection_string, parse_mask_string uses `continue`
        instead of raising — bad tokens are silently ignored.
        """
        result = parse_mask_string("1-3, abc, 5")
        assert 1 in result and 5 in result
        # 'abc' should have been silently skipped

    def test_result_is_sorted(self):
        result = parse_mask_string("10, 1-3")
        assert result == sorted(result)

    def test_duplicates_removed(self):
        result = parse_mask_string("1-3, 2-4")
        assert result == sorted(set(result))


# =============================================================================
# 5. save_and_reload_config
# =============================================================================

class TestSaveAndReloadConfig:
    """
    Verify that save_and_reload_config correctly writes a namespace to
    disk and that the file can be re-loaded via parse_with_config.
    """

    def test_round_trip_without_parser(self, tmp_path):
        """
        Write a namespace with save_and_reload_config (no parser),
        then read it back via parse_with_config.
        """
        ns = argparse.Namespace(cutoff=12.5, representation="cb", n_layers=5, config=None)
        out_path = str(tmp_path / "saved.cfg")
        save_and_reload_config(ns, out_path, parser=None)

        # Verify file was created
        assert os.path.exists(out_path), "Config file not written."

        # Read it back
        parser = _make_parser()
        with patch("sys.argv", ["prog", "--config", out_path]):
            reloaded = parse_with_config(parser)

        assert reloaded.cutoff == 12.5
        assert reloaded.representation == "cb"
        assert reloaded.n_layers == 5

    def test_round_trip_with_parser(self, tmp_path):
        """
        Write a namespace using the parser-grouped path, then read back.
        """
        ns = argparse.Namespace(
            config=None, cutoff=8.0, representation="com",
            n_layers=2, flag=True,
        )
        parser = _make_parser()
        out_path = str(tmp_path / "grouped.cfg")
        save_and_reload_config(ns, out_path, parser=parser)

        with patch("sys.argv", ["prog", "--config", out_path]):
            parser2 = _make_parser()
            reloaded = parse_with_config(parser2)

        assert reloaded.cutoff == 8.0
        assert reloaded.representation == "com"
        assert reloaded.n_layers == 2

    def test_excludes_debug_keys(self, tmp_path):
        """
        Keys like 'debug_help' must not appear in the saved file.
        """
        ns = argparse.Namespace(
            config=None, cutoff=10.0, debug_help=True,
            representation="ca", n_layers=3, flag=False,
        )
        out_path = str(tmp_path / "out.cfg")
        save_and_reload_config(ns, out_path, parser=None)

        content = open(out_path).read()
        assert "debug_help" not in content, (
            "debug_help must be excluded from the saved config."
        )

    def test_returns_namespace(self, tmp_path):
        """save_and_reload_config must return the original namespace."""
        ns = argparse.Namespace(cutoff=10.0, config=None)
        out_path = str(tmp_path / "out.cfg")
        result = save_and_reload_config(ns, out_path, parser=None)
        assert result is ns
