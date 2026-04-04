"""Unit tests for brain_surgery.main interactive bootstrap."""

import runpy
import sys
from types import ModuleType
from typing import ClassVar

import pytest

import brain_surgery.main as main_module


class FakeWrapper:
    """Simple stand-in for ModelWrapper construction checks."""

    init_calls: ClassVar[int] = 0

    def __init__(self, model_name: str, layer_idx: int) -> None:
        _ = model_name
        _ = layer_idx
        type(self).init_calls += 1


def test_main_initializes_wrapper_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify main() initializes wrapper once and reuses it on second call."""
    FakeWrapper.init_calls = 0
    main_module.wrapper = None
    monkeypatch.setattr(main_module, "ModelWrapper", FakeWrapper)

    main_module.main()
    main_module.main()

    assert FakeWrapper.init_calls == 1


def test_main_module_runs_as_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify __main__ guard path executes without loading real model weights."""
    fake_mod = ModuleType("brain_surgery.model_wrapper")

    class ScriptWrapper:
        def __init__(self, model_name: str, layer_idx: int) -> None:
            _ = model_name
            _ = layer_idx

    fake_mod.ModelWrapper = ScriptWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "brain_surgery.model_wrapper", fake_mod)

    runpy.run_module("brain_surgery.main", run_name="__main__")
