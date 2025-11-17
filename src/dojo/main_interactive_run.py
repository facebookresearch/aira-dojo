"""
Script to manually interact with a task environment for AIRS Bench tasks

This script allows vendors to:
1. Configure and load an AIRS Bench task from aira-bench-data
2. Execute their solution code in an Apptainer container
3. Generate a submission.csv file that can be graded by grade_code.py

Note: Uses imports from facebookresearch/aira-dojo (public repo)
"""
import json
from pathlib import Path
from dojo.config_dataclasses.task import TASK_MAP # this is still mlebench unless we add airabench to __init__.py
from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
from dojo.utils.config import build
from dojo.config_dataclasses.task.airabench import AiraBenchTaskConfig
from dojo.config_dataclasses.interpreter.jupyter import JupyterInterpreterConfig
from omegaconf import II
import tempfile

# ===== CONFIGURATION =====

# TODO: Path to the task folder in aira-bench-data (make sure this is mapped in Apptainer)
# Example: "{path_to_aira-bench-data_repo}/aira-bench-data/text_only_tasks/rad/batch_3/TextClassificationWnliCaAccuracy"
TASK_FOLDER_PATH = "/home/munaaghamelu/AnnotationTask/aira-bench-data/text_only_tasks/rad/batch_4/TimeSeriesForecastingCovidDeathsMAE"

# Path to the data directory where the vendor will write outputs
# This is where submission.csv will be created
DATA_DIR = "/home/munaaghamelu/AnnotationTask/aira-bench-data/datasets/airs_text_vendor"

# Rather than global shared datasets directory, use the dataset directory accessible to the vendor
# Path to global shared datasets (used by prepare.py)
GLOBAL_SHARED_DATA_DIR = "/home/munaaghamelu/AnnotationTask/aira-bench-data/datasets/airs_text_vendor"

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
eval_interpreter = build(interpreter_config, INTERPRETER_MAP, data_dir=task_config.eval_data_dir)

# ===== PREPARE THE TASK =====

print("Preparing task environment...")
state, task_info = task.prepare(solver_interpreter=interpreter, eval_interpreter=eval_interpreter)
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

train_ds = load_from_disk("./data/train")  # contains fields: target, label_target
test_ds = load_from_disk("./data/test")  # contains fields: target, (no label_target)

assert len(test_ds) == 266, f"Expected 266 test series, got {len(test_ds)}"

HORIZON = 30  # number of days to predict
forecasts = []  # list of JSON encoded forecast strings

for idx, hist in enumerate(test_ds["target"]):
    series = np.asarray(hist, dtype=np.float64)

    if series.size == 0:
        pred = np.zeros(HORIZON)
    elif series.size == 1:
        pred = np.full(HORIZON, series[-1])
    else:
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None, damped_trend=False
            )
            fit = model.fit(optimized=True, use_brute=True)
            pred = fit.forecast(HORIZON)
        except Exception as e:
            # If the model fails (e.g., numerical issues), fall back to naive.
            pred = np.full(HORIZON, series[-1])

    last_observed = series[-1]
    pred = np.maximum.accumulate(np.concatenate(([last_observed], pred)))[1:]

    forecasts.append(json.dumps(pred.tolist()))

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(test_ds)} series")

submission_df = pd.DataFrame({"label_target": forecasts})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)
"""

# ===== EXECUTE THE SOLUTION =====

print("\n" + "="*60)
print("Executing solution code...")
print("="*60 + "\n")

state, outcome = task.step_task(state, my_solution)

print("\n" + "="*60)
print("Execution Results")
print("="*60)

# outcome is a dictionary, not an object
exec_output = outcome.get('execution_output')
if exec_output:
    print(f"Success: {exec_output.exit_code == 0}")
    print(f"Exit Code: {exec_output.exit_code}")
    print(f"Timed Out: {exec_output.timed_out}")
    print(f"Execution Time: {exec_output.exec_time:.2f}s")

    print("\n--- Standard Output ---")
    for line in exec_output.term_out:
        print(line)

    if hasattr(exec_output, "stderr") and exec_output.stderr:
        print("\n--- Standard Error ---")
        for line in exec_output.term_err:
            print(line)

# ===== EVALUATE THE SOLUTION =====

# The evaluation happens automatically
# Check if we have evaluation results in the outcome
if 'validation_fitness' in outcome:
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print(f"Validation Fitness: {outcome['validation_fitness']}")

if 'test_fitness' in outcome:
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Test Fitness Score: {outcome['test_fitness']}")

if 'aux_eval_info' in outcome:
    print(f"Evaluation Info: {outcome['aux_eval_info']}")

if 'valid_solution' in outcome:
    print(f"Valid Solution: {outcome['valid_solution']}")
    if 'valid_solution_feedback' in outcome:
        print(f"Feedback: {outcome['valid_solution_feedback']}")

# ===== CLEANUP =====

print("\n" + "="*60)
print("Cleaning up...")
print("="*60)
task.close(state)
print("Done!")

# ===== GRADE THE SOLUTION =====

print("\n" + "="*60)
print("Grading the Solution")
print("="*60)

from dojo.core.tasks.constants import EXECUTION_OUTPUT, TEST_FITNESS, VALID_SOLUTION, VALIDATION_FITNESS
tmp_dir = tempfile.mkdtemp(prefix="rad_results_")

exec_output = outcome.get(EXECUTION_OUTPUT)

summary = {
    "success": False,
    "exit_code": None,
    "timed_out": False,
    "exec_time": None,
    # validation/test/valid flags are stored in the outcome dict, not on ExecutionResult
    "valid_solution": outcome.get(VALID_SOLUTION, False),
    "validation_fitness": outcome.get(VALIDATION_FITNESS),
    "test_fitness": outcome.get(TEST_FITNESS),
    "error_output": [],
    "stdout": [],
}

if exec_output is not None:
    summary["success"] = exec_output.exit_code == 0 and not exec_output.timed_out
    summary["exit_code"] = exec_output.exit_code
    summary["timed_out"] = exec_output.timed_out
    summary["exec_time"] = exec_output.exec_time
    # ExecutionResult exposes term_out; term_err may not exist on all implementations
    summary["error_output"] = getattr(exec_output, "term_err", []) or []
    summary["stdout"] = getattr(exec_output, "term_out", []) or []

    grading_report_path = Path(tmp_dir) / "grading_report.json"
    if grading_report_path.exists():
        with open(grading_report_path, "r") as f:
            grading_report = json.load(f)
        summary["grading_report"] = grading_report

print("Grading Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")

