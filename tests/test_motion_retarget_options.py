from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_retarget_options import resolve_max_iter


def test_resolve_max_iter_defaults_to_current_solver_limit() -> None:
    assert resolve_max_iter(None) == 10


def test_resolve_max_iter_accepts_positive_override() -> None:
    assert resolve_max_iter(20) == 20


def test_resolve_max_iter_rejects_non_positive_values() -> None:
    try:
        resolve_max_iter(0)
    except ValueError as exc:
        assert "max_iter" in str(exc)
    else:
        raise AssertionError("Expected non-positive max_iter to be rejected")
