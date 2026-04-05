"""Unit tests for script path handling and checkpoint wiring."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
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


def test_verify_pilot_pick_dynamic_elbow_k_detects_slowdown() -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_elbow", SCRIPTS_DIR / "verify_pilot.py"
    )

    best_k = verify_pilot._pick_dynamic_elbow_k(
        k_values=[4, 6, 8, 10, 12],
        inertias=[1000.0, 700.0, 580.0, 520.0, 490.0],
    )

    assert best_k == 6


def test_verify_pilot_run_phase_q4_q5_uses_spherical_features_and_writes_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_spherical", SCRIPTS_DIR / "verify_pilot.py"
    )

    class FakeInterpreter:
        def __init__(self) -> None:
            self.latents = torch.tensor(
                [
                    [5.0, 0.1, 0.0],
                    [4.0, 1.0, 0.0],
                    [0.5, 5.0, 0.0],
                    [0.1, 4.0, 6.0],
                ],
                dtype=torch.float32,
            )
            self.metadata = [
                {"category": "Clubs"},
                {"category": "Clubs"},
                {"category": "Historical"},
                {"category": "Tactical"},
            ]

        activation_matrix = torch.randn(3, 4)

        def rank_features_by_max_activation(
            self, top_k: int = 10
        ) -> list[dict[str, object]]:
            _ = top_k
            return [
                {
                    "rank": 1,
                    "feature_index": 0,
                    "max_feature_value": 5.0,
                }
            ]

        def get_top_examples_for_feature(
            self, feature_index: int, top_k: int = 10
        ) -> list[dict[str, object]]:
            _ = top_k
            return [
                {
                    "token_text": f"token_{feature_index}",
                    "feature_value": 1.0,
                }
            ]

    captured: dict[str, object] = {}

    def fake_elbow(
        feature_profiles: np.ndarray,
        *,
        start_k: int,
        step: int,
        max_k: int,
    ) -> tuple[int, list[dict[str, object]]]:
        _ = (start_k, step, max_k)
        captured["norms"] = np.linalg.norm(feature_profiles, axis=1)
        return 2, []

    def fake_cluster(
        interpreter: object,
        num_clusters: int,
        random_state: int,
        feature_profiles: np.ndarray | None = None,
    ) -> dict[str, object]:
        _ = (interpreter, num_clusters, random_state)
        captured["feature_profiles"] = feature_profiles
        return {
            "cluster_summaries": [
                {
                    "cluster_id": 0,
                    "num_features": 2,
                    "feature_indices": [0, 1],
                    "representative_feature": 0,
                    "representative_tokens": ["club"],
                },
                {
                    "cluster_id": 1,
                    "num_features": 1,
                    "feature_indices": [2],
                    "representative_feature": 2,
                    "representative_tokens": ["tactic"],
                },
            ]
        }

    monkeypatch.setattr(verify_pilot, "_print_elbow_table", fake_elbow)
    monkeypatch.setattr(verify_pilot, "cluster_features_kmeans", fake_cluster)

    cluster_report = tmp_path / "cluster_report.json"
    verify_pilot.run_phase_q4_q5(
        FakeInterpreter(),
        elbow_start_k=4,
        elbow_step=2,
        elbow_max_k=8,
        cluster_report_json_path=cluster_report,
    )

    norms = captured.get("norms")
    assert isinstance(norms, np.ndarray)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)

    payload = json.loads(cluster_report.read_text(encoding="utf-8"))
    assert payload["selected_k"] == 2
    assert payload["theme_summary"]["Clubs"] == 2
    assert payload["theme_summary"]["Historical"] == 0
    assert payload["theme_summary"]["Tactical"] == 0


def test_verify_pilot_experiment_elbow_flag_applies_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_pilot = _load_script_module(
        "verify_pilot_test_experiment_flag",
        SCRIPTS_DIR / "verify_pilot.py",
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

    def fake_run_phase_q4_q5(
        interpreter: object,
        *,
        elbow_start_k: int,
        elbow_step: int,
        elbow_max_k: int,
        top_features_csv_path: Path | None = None,
        elbow_json_path: Path | None = None,
        elbow_plot_path: Path | None = None,
        cluster_report_json_path: Path | None = None,
        global_census_csv_path: Path | None = None,
    ) -> None:
        _ = (
            interpreter,
            top_features_csv_path,
            elbow_json_path,
            elbow_plot_path,
            cluster_report_json_path,
            global_census_csv_path,
        )
        captured["elbow_start_k"] = elbow_start_k
        captured["elbow_step"] = elbow_step
        captured["elbow_max_k"] = elbow_max_k

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_pilot.py",
            "--experiment-elbow",
        ],
    )
    monkeypatch.setattr(verify_pilot, "_validate_default_files", lambda *_: None)
    monkeypatch.setattr(verify_pilot, "SAEInterpreter", FakeInterpreter)
    monkeypatch.setattr(verify_pilot, "ModelWrapper", FakeModelWrapper)
    monkeypatch.setattr(verify_pilot, "run_phase_q4_q5", fake_run_phase_q4_q5)
    monkeypatch.setattr(
        verify_pilot,
        "run_phase_q6",
        lambda *args, **kwargs: ({}, {}, {}, object()),
    )
    monkeypatch.setattr(verify_pilot, "run_dtype_audit", lambda *args, **kwargs: None)

    verify_pilot.main()

    assert captured["elbow_start_k"] == verify_pilot.EXPERIMENT_ELBOW_START_K
    assert captured["elbow_step"] == verify_pilot.EXPERIMENT_ELBOW_STEP
    assert captured["elbow_max_k"] == verify_pilot.EXPERIMENT_ELBOW_MAX_K


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
