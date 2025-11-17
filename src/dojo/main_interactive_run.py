"""
Script to manually interact with a task environment for AIRS Bench tasks

This script allows vendors to:
1. Configure and load an AIRS Bench task from aira-bench-data
2. Execute their solution code in an Apptainer container
3. Generate a submission.csv file that can be graded by grade_code.py

Note: Uses imports from facebookresearch/aira-dojo (public repo)
"""
from pathlib import Path
from dojo.config_dataclasses.task import TASK_MAP # this is still mlebench unless we add airabench to __init__.py
from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
from dojo.utils.config import build
from dojo.config_dataclasses.task.airabench import AiraBenchTaskConfig
from dojo.config_dataclasses.interpreter.jupyter import JupyterInterpreterConfig
from omegaconf import II

# ===== CONFIGURATION =====

# TODO: Path to the task folder in aira-bench-data (make sure this is mapped in Apptainer)
# Example: "{path_to_aira-bench-data_repo}/aira-bench-data/text_only_tasks/rad/batch_3/TextClassificationWnliCaAccuracy"
TASK_FOLDER_PATH = "~/AnnotationTask/aira-bench-data/text_only_tasks/rad/batch_4/TimeSeriesForecastingCovidDeathsMAE"

# Path to the data directory where the vendor will write outputs
# This is where submission.csv will be created
DATA_DIR = II("get_tmp_dir:train_data")

# Rather than global shared datasets directory, use the dataset directory accessible to the vendor
# Path to global shared datasets (used by prepare.py)
GLOBAL_SHARED_DATA_DIR = "~/AnnotationTask/aira-bench-data/datasets/airs_text_vendor"

# Apptainer superimage version (use the version shipped with MLEBench)
SUPERIMAGE_VERSION = "2025-05-02v2"

# Timeout in seconds (2 hours)
TIMEOUT = 7200

# ===== TASK CONFIGURATION =====

task_config = AiraBenchTaskConfig(
    name=Path(TASK_FOLDER_PATH).name,  # Extract task name from folder path
    benchmark="airabench",
    data_dir=DATA_DIR,  # Where the agent can read/write data
    eval_data_dir=DATA_DIR,  # Same as data_dir for evaluation
    global_shared_data_dir=GLOBAL_SHARED_DATA_DIR,  # Raw datasets location
    eval_every_step=True,  # Evaluate after each step
)

# ===== INTERPRETER CONFIGURATION =====

interpreter_config = JupyterInterpreterConfig(
    superimage_version=SUPERIMAGE_VERSION,
    timeout=TIMEOUT,
    strip_ansi=True,  # Clean up output
)

# ===== BUILD TASK AND INTERPRETER =====

task = build(task_config, TASK_MAP)
interpreter = build(interpreter_config, INTERPRETER_MAP, data_dir=task_config.data_dir)

# ===== PREPARE THE TASK =====

print("Preparing task environment...")
state, task_info = task.prepare(solver_interpreter=interpreter, eval_interpreter=None)
print("Task environment ready!")
print(f"Task info: {task_info}")

# ===== YOUR SOLUTION CODE =====

# For AIRS Bench tasks, your solution should:
# 1. Load data from ./data directory (created by prepare.py in the container)
#    - For text tasks: typically train and test datasets as HuggingFace datasets
#    - Access using: load_from_disk('./data/train') and load_from_disk('./data/test')
# 2. Train your model on the training data
# 3. Generate predictions on the test set
# 4. Save predictions to ./submission.csv in the format expected by evaluate.py
#
# Example for a time series task: (taken from past log)
my_solution = """

import json
import numpy as np
import pandas as pd
from datasets import load_from_disk
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
# The repository provides the data under ``./data/train`` and ``./data/test``.
# ``train`` is not needed for this simple baseline, but we load it anyway
# to illustrate the typical workflow.
train_ds = load_from_disk("./data/train")  # contains fields: target, label_target
test_ds = load_from_disk("./data/test")  # contains fields: target, (no label_target)

# Verify we have the expected number of series
assert len(test_ds) == 266, f"Expected 266 test series, got {len(test_ds)}"

# ----------------------------------------------------------------------
# 2. Forecast each series
# ----------------------------------------------------------------------
HORIZON = 30  # number of days to predict
forecasts = []  # list of JSON‑encoded forecast strings

for idx, hist in enumerate(test_ds["target"]):
    # ``hist`` is a Python list of floats (historical cumulative deaths)
    series = np.asarray(hist, dtype=np.float64)

    # ------------------------------------------------------------------
    # Edge cases: very short series or constant series
    # ------------------------------------------------------------------
    if series.size == 0:
        # No history – predict zeros
        pred = np.zeros(HORIZON)
    elif series.size == 1:
        # Only one point – repeat the last observed value
        pred = np.full(HORIZON, series[-1])
    else:
        # --------------------------------------------------------------
        # Fit a simple additive Holt linear trend model
        # --------------------------------------------------------------
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None, damped_trend=False
            )
            fit = model.fit(optimized=True, use_brute=True)
            pred = fit.forecast(HORIZON)
        except Exception as e:
            # If the model fails (e.g., numerical issues), fall back to naive.
            pred = np.full(HORIZON, series[-1])

    # --------------------------------------------------------------
    # Post‑process: enforce monotonicity (cumulative nature)
    # --------------------------------------------------------------
    # Ensure the forecast never goes below the last observed cumulative count.
    last_observed = series[-1]
    pred = np.maximum.accumulate(np.concatenate(([last_observed], pred)))[1:]

    # Convert to plain Python list and JSON‑encode
    forecasts.append(json.dumps(pred.tolist()))

    # Optional: progress indicator
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(test_ds)} series")

# ----------------------------------------------------------------------
# 3. Write submission file
# ----------------------------------------------------------------------
submission_df = pd.DataFrame({"label_target": forecasts})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file written to: {submission_path}")
print(f"Shape: {submission_df.shape} (rows, columns)")
print("First few rows:")
print(submission_df.head())
"""

# Query - Do we want this solution to be parsed from RAD output? So an agent's acutual attempt of a task?
# ===== EXECUTE THE SOLUTION =====

print("\n" + "="*60)
print("Executing solution code...")
print("="*60 + "\n")

state, outcome = task.step_task(state, my_solution)

print("\n" + "="*60)
print("Execution Results")
print("="*60)
print(f"Success: {outcome.execution_output.exit_code == 0}")
print(f"Exit Code: {outcome.execution_output.exit_code}")
print(f"Timed Out: {outcome.execution_output.timed_out}")
print(f"Execution Time: {outcome.execution_output.exec_time:.2f}s")

print("\n--- Standard Output ---")
for line in outcome.execution_output.term_out:
    print(line)

if outcome.execution_output.term_err:
    print("\n--- Standard Error ---")
    for line in outcome.execution_output.term_err:
        print(line)

# ===== EVALUATE THE SOLUTION =====

# The evaluation happens automatically due to eval_every_step=True
# Check if we have evaluation results in the outcome
if hasattr(outcome, 'fitness') and outcome.fitness is not None:
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Fitness Score: {outcome.fitness}")

if hasattr(outcome, 'grading_report') and outcome.grading_report is not None:
    print(f"Grading Report: {outcome.grading_report}")

# ===== CLEANUP =====

print("\n" + "="*60)
print("Cleaning up...")
print("="*60)
task.close(state)
print("Done!")

# ===== NEXT STEPS =====

print("\n" + "="*60)
print("Next Steps")
print("="*60)
print(f"1. Check submission file at: {DATA_DIR}/submission.csv")
print(f"2. You can now run grade_code.py on the submission:")
print(f"   python grade_code.py {DATA_DIR}/submission.csv")
print("="*60)