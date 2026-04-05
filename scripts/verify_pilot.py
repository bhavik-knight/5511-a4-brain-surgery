"""Pilot verification script for Q4/Q5/Q6 with dtype audit.

Run with:
    uv run scripts/verify_pilot.py
"""

import argparse
from pathlib import Path

import numpy as np

from brain_surgery import pilot_pipeline
from brain_surgery.clustering import cluster_features_kmeans
from brain_surgery.interpret import SAEInterpreter
from brain_surgery.intervention import SAEIntervention
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.utils import (
    ACTIVATIONS_DIR,
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER_IDX,
    EXPERIMENTS_DIR,
    create_run_output_dirs,
    generate_run_id,
)

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "sae_checkpoint.pt"
DEFAULT_DATASET = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
DEFAULT_ELBOW_START_K = 4
DEFAULT_ELBOW_STEP = 2
DEFAULT_ELBOW_MAX_K = 20
EXPERIMENT_ELBOW_START_K = 10
EXPERIMENT_ELBOW_STEP = 5
EXPERIMENT_ELBOW_MAX_K = 100


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_header(title: str) -> None:
    """Public wrapper for section headers used by companion scripts."""
    _print_header(title)


def _validate_default_files(checkpoint_path: Path, dataset_path: Path) -> None:
    checkpoint_local = Path(checkpoint_path).expanduser()
    dataset_local = Path(dataset_path).expanduser()

    missing: list[Path] = []
    if not checkpoint_local.exists():
        missing.append(checkpoint_local)
    if not dataset_local.exists():
        missing.append(dataset_local)

    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required pilot artifacts are missing (paths checked):\n"
            f"{lines}\n"
            "Generate them first (golden run / training pipeline), then retry."
        )


def validate_default_files(checkpoint_path: Path, dataset_path: Path) -> None:
    """Public wrapper for artifact-path validation used by companion scripts."""
    _validate_default_files(checkpoint_path, dataset_path)


def _resolve_checkpoint_path(args: argparse.Namespace, run_id: str) -> Path:
    """Resolve checkpoint path with run-aware fallback logic.

    Priority:
    1. Explicit `--checkpoint` if provided.
    2. `results/experiments/<run_id>/checkpoints/sae_best.pt` if it exists.
    3. Global default checkpoint path.
    """
    if args.checkpoint is not None:
        return Path(args.checkpoint)

    run_checkpoint = EXPERIMENTS_DIR / run_id / "checkpoints" / "sae_best.pt"
    if run_checkpoint.exists():
        return run_checkpoint

    return DEFAULT_CHECKPOINT


def _safe_token_text(value: object) -> str:
    return pilot_pipeline.safe_token_text(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pilot verification with run-scoped output artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to activation dataset (.pt).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(DEFAULT_MODEL_NAME),
        help="Local directory containing model weights.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. If omitted, generates run_YYYYMMDD_HHMM.",
    )
    parser.add_argument(
        "--elbow-start-k",
        type=int,
        default=DEFAULT_ELBOW_START_K,
        help="Starting k for elbow sweep.",
    )
    parser.add_argument(
        "--elbow-step",
        type=int,
        default=DEFAULT_ELBOW_STEP,
        help="Step size for elbow sweep.",
    )
    parser.add_argument(
        "--elbow-max-k",
        type=int,
        default=DEFAULT_ELBOW_MAX_K,
        help="Maximum k for elbow sweep.",
    )
    parser.add_argument(
        "--experiment-elbow",
        action="store_true",
        help=(
            "Use experiment sweep preset "
            f"({EXPERIMENT_ELBOW_START_K}, {EXPERIMENT_ELBOW_STEP}, "
            f"{EXPERIMENT_ELBOW_MAX_K})."
        ),
    )
    parser.add_argument(
        "--target-feature",
        type=int,
        default=1625,
        help="Latent feature index to clamp for Q6 target intervention.",
    )
    parser.add_argument(
        "--control-feature",
        type=int,
        default=0,
        help="Latent feature index used as Q6 control intervention.",
    )
    parser.add_argument(
        "--clamp-value",
        type=float,
        default=8.0,
        help="Clamp multiplier applied to feature max value for Q6.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who is the better football player, Ronaldo or Messi?",
        help="Prompt used for Q6 intervention scoring.",
    )
    parser.add_argument(
        "--candidate-tokens",
        type=str,
        nargs="+",
        default=["Ronaldo", "Messi", "football", "goal"],
        help="Candidate next tokens scored during Q6 intervention.",
    )
    return parser.parse_args()


def run_phase_q4_q5(
    interpreter: SAEInterpreter,
    *,
    elbow_start_k: int = DEFAULT_ELBOW_START_K,
    elbow_step: int = DEFAULT_ELBOW_STEP,
    elbow_max_k: int = DEFAULT_ELBOW_MAX_K,
    top_features_csv_path: Path | None = None,
    elbow_json_path: Path | None = None,
    elbow_plot_path: Path | None = None,
    cluster_report_json_path: Path | None = None,
    global_census_csv_path: Path | None = None,
) -> None:
    """Run feature interpretation and clustering summary for Q4/Q5."""
    pilot_pipeline.run_phase_q4_q5(
        interpreter,
        elbow_start_k=elbow_start_k,
        elbow_step=elbow_step,
        elbow_max_k=elbow_max_k,
        top_features_csv_path=top_features_csv_path,
        elbow_json_path=elbow_json_path,
        elbow_plot_path=elbow_plot_path,
        cluster_report_json_path=cluster_report_json_path,
        global_census_csv_path=global_census_csv_path,
        print_header_fn=_print_header,
        safe_token_text_fn=_safe_token_text,
        print_elbow_table_fn=_print_elbow_table,
        save_top_features_csv_fn=pilot_pipeline.save_top_features_csv,
        save_elbow_sweep_json_fn=pilot_pipeline.save_elbow_sweep_json,
        save_elbow_plot_fn=pilot_pipeline.save_elbow_plot,
        save_cluster_report_fn=pilot_pipeline.save_cluster_report,
        cluster_features_kmeans_fn=cluster_features_kmeans,
    )


def _print_elbow_table(
    feature_profiles: np.ndarray,
    *,
    start_k: int,
    step: int,
    max_k: int,
) -> tuple[int, list[dict[str, object]]]:
    """Print SSE sweep and return automatically selected elbow k."""
    return pilot_pipeline.print_elbow_table(
        feature_profiles,
        start_k=start_k,
        step=step,
        max_k=max_k,
        pick_elbow_k_fn=_pick_dynamic_elbow_k,
    )


def _pick_dynamic_elbow_k(k_values: list[int], inertias: list[float]) -> int:
    """Pick k where SSE improvement rate slows down significantly.

    Uses the discrete rate of change of SSE improvements and chooses the first
    k whose slowdown exceeds a dynamic threshold derived from the observed sweep.
    """
    return pilot_pipeline.pick_dynamic_elbow_k(k_values, inertias)


def run_phase_q6(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    *,
    checkpoint_path: Path,
    prompt: str = "Who is the better football player, Ronaldo or Messi?",
    candidate_tokens: list[str] | None = None,
    target_feature_index: int = 1625,
    control_feature_index: int = 0,
    clamp_multiplier: float = 8.0,
    intervention_csv_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], SAEIntervention]:
    """Run Ronaldo intervention and print baseline vs clamped deltas for Q6."""
    return pilot_pipeline.run_phase_q6(
        interpreter,
        model_wrapper,
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        candidate_tokens=candidate_tokens,
        target_feature_index=target_feature_index,
        control_feature_index=control_feature_index,
        clamp_multiplier=clamp_multiplier,
        intervention_csv_path=intervention_csv_path,
        print_header_fn=_print_header,
        save_intervention_csv_fn=pilot_pipeline.save_intervention_csv,
        intervention_cls=SAEIntervention,
    )


def run_dtype_audit(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    intervention: SAEIntervention,
    *,
    metadata_json_path: Path,
) -> None:
    """Print dtype audit and confirm rich metadata report export."""
    pilot_pipeline.run_dtype_audit(
        interpreter,
        model_wrapper,
        intervention,
        metadata_json_path=metadata_json_path,
        print_header_fn=_print_header,
        save_metadata_report_fn=pilot_pipeline.save_metadata_report,
    )


def main() -> None:
    """Run full pilot verification report for Q4/Q5/Q6."""
    args = _parse_args()
    if args.experiment_elbow:
        args.elbow_start_k = EXPERIMENT_ELBOW_START_K
        args.elbow_step = EXPERIMENT_ELBOW_STEP
        args.elbow_max_k = EXPERIMENT_ELBOW_MAX_K

    run_id = args.run_id or generate_run_id()
    checkpoint_path = _resolve_checkpoint_path(args, run_id)
    dataset_path = args.dataset
    model_dir = str(args.model_dir)
    run_dirs = create_run_output_dirs(run_id)
    top_features_csv_path = run_dirs["features_run"] / f"top_10_features_{run_id}.csv"
    elbow_json_path = run_dirs["metrics_root"] / f"elbow_sweep_{run_id}.json"
    elbow_plot_path = run_dirs["metrics_root"] / f"elbow_plot_{run_id}.png"
    cluster_report_json_path = (
        run_dirs["experiment_root"] / f"cluster_report_{run_id}.json"
    )
    intervention_csv_path = (
        run_dirs["experiment_root"] / f"intervention_results_{run_id}.csv"
    )
    metadata_json_path = run_dirs["experiment_root"] / f"metadata_{run_id}.json"
    global_census_csv_path = (
        run_dirs["features_run"] / f"global_feature_census_{run_id}.csv"
    )

    _print_header("Pilot Verification Report")
    print(f"Run ID:     {run_id}")
    print(f"Run dir:    {run_dirs['root']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Model dir:  {model_dir}")

    _validate_default_files(checkpoint_path, dataset_path)

    interpreter = SAEInterpreter(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device="cpu",
    )
    interpreter.load()
    interpreter.compute_latents()

    run_phase_q4_q5(
        interpreter,
        elbow_start_k=args.elbow_start_k,
        elbow_step=args.elbow_step,
        elbow_max_k=args.elbow_max_k,
        top_features_csv_path=top_features_csv_path,
        elbow_json_path=elbow_json_path,
        elbow_plot_path=elbow_plot_path,
        cluster_report_json_path=cluster_report_json_path,
        global_census_csv_path=global_census_csv_path,
    )

    model_wrapper = ModelWrapper(model_name=model_dir, layer_idx=DEFAULT_LAYER_IDX)
    _, _, _, intervention = run_phase_q6(
        interpreter,
        model_wrapper,
        checkpoint_path=checkpoint_path,
        prompt=args.prompt,
        candidate_tokens=args.candidate_tokens,
        target_feature_index=args.target_feature,
        control_feature_index=args.control_feature,
        clamp_multiplier=args.clamp_value,
        intervention_csv_path=intervention_csv_path,
    )

    run_dtype_audit(
        interpreter,
        model_wrapper,
        intervention,
        metadata_json_path=metadata_json_path,
    )

    _print_header("Verification Complete")
    print("All requested pilot checks completed successfully.")


if __name__ == "__main__":
    main()
