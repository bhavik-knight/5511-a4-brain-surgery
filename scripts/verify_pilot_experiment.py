"""Experimental pilot verification settings for server-side clustering sweeps.

This script reuses the core pilot workflow but runs elbow diagnostics with:
- start_k=10
- step=5
- max_k=100

Run with:
    uv run scripts/verify_pilot_experiment.py
"""

from verify_pilot import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    _print_header,
    _validate_default_files,
    run_dtype_audit,
    run_phase_q4_q5,
    run_phase_q6,
)
from brain_surgery.interpret import SAEInterpreter
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.utils import (
    DEFAULT_MODEL_NAME,
    create_run_output_dirs,
    generate_run_id,
)

EXPERIMENT_ELBOW_START_K = 10
EXPERIMENT_ELBOW_STEP = 5
EXPERIMENT_ELBOW_MAX_K = 100


def main() -> None:
    """Run pilot verification with experimental elbow sweep settings."""
    checkpoint_path = DEFAULT_CHECKPOINT
    dataset_path = DEFAULT_DATASET
    run_id = generate_run_id()
    run_dirs = create_run_output_dirs(f"experiment_{run_id}")

    _print_header("Pilot Verification Report (Experiment)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Model dir:  {DEFAULT_MODEL_NAME}")
    print(
        "Elbow config: "
        f"start_k={EXPERIMENT_ELBOW_START_K}, "
        f"step={EXPERIMENT_ELBOW_STEP}, "
        f"max_k={EXPERIMENT_ELBOW_MAX_K}"
    )

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
        elbow_start_k=EXPERIMENT_ELBOW_START_K,
        elbow_step=EXPERIMENT_ELBOW_STEP,
        elbow_max_k=EXPERIMENT_ELBOW_MAX_K,
    )

    model_wrapper = ModelWrapper(model_name=DEFAULT_MODEL_NAME, layer_idx=12)
    _, _, _, intervention = run_phase_q6(interpreter, model_wrapper)

    run_dtype_audit(
        interpreter,
        model_wrapper,
        intervention,
        metadata_json_path=run_dirs["root"] / "metadata.json",
    )

    _print_header("Experiment Verification Complete")
    print("All experiment checks completed successfully.")


if __name__ == "__main__":
    main()
