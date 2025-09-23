import json
import sys
from pathlib import Path
from typing import Any, Dict

from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
from dojo.config_dataclasses.interpreter.jupyter import JupyterInterpreterConfig
from dojo.config_dataclasses.task import TASK_MAP
from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig
from dojo.utils.config import build
from dojo.utils.environment import get_mlebench_data_dir

SUPERIMAGE_VERSION = "ENTER YOUR SUPERIMAGE FILE NAME HERE"

def execute_code(code_file_path: str, task_name: str) -> Dict[str, Any]:
    code_path = Path(code_file_path)
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found: {code_file_path}")

    code_content = code_path.read_text()
    cache_dir = get_mlebench_data_dir()
    results_output_dir = "/tmp/rad_results"

    task_config = MLEBenchTaskConfig(
        name=str(task_name),
        benchmark="mlebench",
        cache_dir=str(cache_dir),
        results_output_dir=str(results_output_dir),
        public_dir=str(f"{cache_dir}/{task_name}/prepared/public"),
        private_dir=str(f"{cache_dir}/{task_name}/prepared/private"),
        data_dir=str(f"{cache_dir}/{task_name}/prepared/public/"),
        submission_fname="submission.csv",
    )

    interpreter_config = JupyterInterpreterConfig(
        superimage_version="SUPERIMAGE_VERSION",
        timeout=7200,
    )

    try:
        task = build(task_config, TASK_MAP)
        solver_interpreter = build(interpreter_config, INTERPRETER_MAP, data_dir=task_config.data_dir)

        state, task_info = task.prepare(solver_interpreter=solver_interpreter, eval_interpreter=None)

        updated_state, eval_result = task.step_task(state, code_content)

        from dojo.core.tasks.constants import EXECUTION_OUTPUT, TEST_FITNESS, VALID_SOLUTION, VALIDATION_FITNESS

        exec_output = eval_result.get(EXECUTION_OUTPUT)
        print(exec_output)
        summary = {
            "success": False,
            "exit_code": None,
            "timed_out": False,
            "exec_time": None,
            "valid_solution": eval_result.get(VALID_SOLUTION, False),
            "validation_fitness": eval_result.get(VALIDATION_FITNESS),
            "test_fitness": eval_result.get(TEST_FITNESS),
            "error_output": [],
            "stdout": [],
        }

        if exec_output is not None:
            summary["success"] = exec_output.exit_code == 0 and not exec_output.timed_out
            summary["exit_code"] = exec_output.exit_code
            summary["timed_out"] = exec_output.timed_out
            summary["exec_time"] = exec_output.exec_time
            summary["error_output"] = exec_output.term_err if hasattr(exec_output, "term_err") else []
            summary["stdout"] = exec_output.term_out if hasattr(exec_output, "term_out") else []

        grading_report_path = Path(results_output_dir) / "grading_report.json"
        if grading_report_path.exists():
            with open(grading_report_path, "r") as f:
                grading_report = json.load(f)
            summary["grading_report"] = grading_report

        return summary

    except Exception as e:
        return {"error": str(e), "success": False}

    finally:
        if "task" in locals() and "state" in locals():
            task.close(state)


def main():
    if len(sys.argv) != 4:
        print("Usage: python grade_code.py <code_file_path> <task_name> <output_path>")
        sys.exit(1)

    code_file_path = sys.argv[1]
    task_name = sys.argv[2]
    output_path = sys.argv[3]

    result = execute_code(code_file_path, task_name)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
