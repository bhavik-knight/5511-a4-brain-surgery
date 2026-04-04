"""Unit tests for script path handling and checkpoint wiring."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def _load_script_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_pilot_resolve_checkpoint_prefers_cli_path() -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_cli", SCRIPTS_DIR / "verify_pilot.py"
    )
    explicit = Path("custom/checkpoint.pt")
    args = argparse.Namespace(checkpoint=explicit)

    resolved = verify_pilot._resolve_checkpoint_path(args, "run_abc")

    assert resolved == explicit


def test_verify_pilot_resolve_checkpoint_uses_run_scoped_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_run", SCRIPTS_DIR / "verify_pilot.py"
    )
    run_id = "run_unit"
    run_checkpoint = tmp_path / "experiments" / run_id / "checkpoints" / "sae_best.pt"
    run_checkpoint.parent.mkdir(parents=True)
    run_checkpoint.write_bytes(b"ok")

    monkeypatch.setattr(verify_pilot, "EXPERIMENTS_DIR", tmp_path / "experiments")
    args = argparse.Namespace(checkpoint=None)

    resolved = verify_pilot._resolve_checkpoint_path(args, run_id)

    assert resolved == run_checkpoint


def test_verify_pilot_validate_default_files_expands_user_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_validate", SCRIPTS_DIR / "verify_pilot.py"
    )
    fake_home = tmp_path / "home"
    dataset = fake_home / "dataset.pt"
    checkpoint = fake_home / "checkpoint.pt"
    fake_home.mkdir(parents=True)
    dataset.write_bytes(b"dataset")
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setenv("HOME", str(fake_home))

    verify_pilot._validate_default_files(Path("~/checkpoint.pt"), Path("~/dataset.pt"))


def test_verify_pilot_run_phase_q6_passes_checkpoint_to_intervention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_q6", SCRIPTS_DIR / "verify_pilot.py"
    )

    captured: dict[str, object] = {}

    class FakeIntervention:
        def __init__(
            self,
            model_wrapper: object,
            checkpoint_path: Path,
            device: str,
        ) -> None:
            _ = model_wrapper
            _ = device
            captured["checkpoint_path"] = checkpoint_path

        def compute_feature_max_values(self, activation_matrix: torch.Tensor) -> None:
            _ = activation_matrix

        def compare_next_token_logprobs(
            self,
            prompt: str,
            candidate_tokens: list[str],
            feature_index: int | None = None,
            clamp_multiplier: float | None = None,
        ) -> dict[str, float]:
            _ = prompt
            _ = feature_index
            _ = clamp_multiplier
            return {token: float(index) for index, token in enumerate(candidate_tokens)}

    fake_interpreter = SimpleNamespace(activation_matrix=torch.randn(3, 4))
    fake_wrapper = object()
    checkpoint = tmp_path / "q6.pt"

    monkeypatch.setattr(verify_pilot, "SAEIntervention", FakeIntervention)

    verify_pilot.run_phase_q6(
        fake_interpreter,
        fake_wrapper,
        checkpoint_path=checkpoint,
    )

    assert captured["checkpoint_path"] == checkpoint


def test_verify_pilot_experiment_forwards_default_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot",
        SCRIPTS_DIR / "verify_pilot.py",
    )
    experiment = _load_script_module(
        "verify_pilot_experiment_test",
        SCRIPTS_DIR / "verify_pilot_experiment.py",
    )

    captured: dict[str, object] = {}

    class FakeInterpreter:
        def __init__(
            self,
            checkpoint_path: Path,
            dataset_path: Path,
            device: str,
        ) -> None:
            _ = checkpoint_path
            _ = dataset_path
            _ = device

        def load(self) -> None:
            return None

        def compute_latents(self) -> None:
            return None

    class FakeModelWrapper:
        def __init__(self, model_name: str, layer_idx: int) -> None:
            _ = model_name
            _ = layer_idx

    def fake_run_phase_q6(
        interpreter: object,
        model_wrapper: object,
        *,
        checkpoint_path: Path,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], object]:
        _ = interpreter
        _ = model_wrapper
        captured["checkpoint_path"] = checkpoint_path
        return {}, {}, {}, object()

    monkeypatch.setattr(experiment, "_validate_default_files", lambda *_: None)
    monkeypatch.setattr(experiment, "SAEInterpreter", FakeInterpreter)
    monkeypatch.setattr(experiment, "ModelWrapper", FakeModelWrapper)
    monkeypatch.setattr(experiment, "run_phase_q4_q5", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "run_phase_q6", fake_run_phase_q6)
    monkeypatch.setattr(experiment, "run_dtype_audit", lambda *args, **kwargs: None)

    experiment.main()

    assert captured["checkpoint_path"] == verify_pilot.DEFAULT_CHECKPOINT


def test_train_university_main_uses_run_scoped_checkpoint_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_university = _load_script_module(
        "train_university_test", SCRIPTS_DIR / "train_university.py"
    )

    run_id = "run_unit"
    run_dirs = {
        "root": tmp_path / "experiments" / run_id,
        "checkpoints": tmp_path / "experiments" / run_id / "checkpoints",
        "logs": tmp_path / "experiments" / run_id / "logs",
        "metrics_root": tmp_path / "metrics",
    }
    for path in run_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        dataset=tmp_path / "dataset.pt",
        checkpoint=None,
        epochs=2,
        batch_size=8,
        lr=1e-4,
        l1=1e-3,
        patience=3,
        resume=False,
        no_wandb=True,
        wandb_project="proj",
        wandb_run_name=None,
        tensorboard_dir=None,
        run_id=run_id,
        smoke_test=False,
    )

    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def train(self, activation_matrix: torch.Tensor) -> tuple[list[float], object]:
            _ = activation_matrix
            return [1.0, 0.5], SimpleNamespace(
                epochs=2,
                final_loss=0.5,
                dead_neuron_fraction=0.1,
                best_loss=0.5,
            )

    monkeypatch.setattr(train_university, "parse_args", lambda: args)
    monkeypatch.setattr(train_university, "create_run_output_dirs", lambda _: run_dirs)
    monkeypatch.setattr(
        train_university,
        "torch",
        SimpleNamespace(
            load=lambda *a, **k: {"activation_matrix": torch.randn(16, 896)},
            cuda=torch.cuda,
            tensor=torch.tensor,
            Tensor=torch.Tensor,
            long=torch.long,
            OutOfMemoryError=torch.cuda.OutOfMemoryError,
        ),
    )
    monkeypatch.setattr(
        train_university,
        "SparseAutoencoder",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(train_university, "SAETrainer", FakeTrainer)

    train_university.main()

    assert captured["checkpoint_path"] == run_dirs["checkpoints"] / "sae_best.pt"
    assert captured["tensorboard_log_dir"] == run_dirs["logs"] / "tensorboard"
