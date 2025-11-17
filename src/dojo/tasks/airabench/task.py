import importlib
import io
import json
import os
import re
import runpy
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from dojo.config_dataclasses.task.airabench import AiraBenchTaskConfig
from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION,
)


@dataclass
class AiraBenchTaskMetadata:
    metric_lower_is_better: bool
    file_export_globs: List[str]
    container_python_requirements: List[str]
    evaluate_container_python_requirements: List[str]
    logging_info: Dict[str, Any]
    prepare_code_python_requirements: Optional[List[str]] = None


@dataclass
class AiraBenchTaskData:
    project_description: str
    prepare_code: str
    metadata: AiraBenchTaskMetadata
    evaluate_prepare_code: str
    evaluate_code: str
    supporting_scripts: List[str] = None  # any other .py files in the folder


def get_aira_bench_task_from_config(config: AiraBenchTaskConfig) -> AiraBenchTaskData:
    """
    Given a task config, finds and loads the AiraBenchTask files.
    Searches for the task folder by name in text_only_tasks.rad (including batch subdirectories).
    """
    # Try to find the task folder by importing the text_only_tasks package and searching
    try:
        import text_only_tasks.rad as rad_module
        # Use __path__ for namespace packages (which may have __file__ = None)
        if hasattr(rad_module, '__path__') and rad_module.__path__:
            rad_path = rad_module.__path__[0]
        else:
            raise RuntimeError("Could not determine rad_module path")
    except ImportError:
        raise RuntimeError("text_only_tasks module not found. Ensure aira-bench-data is installed.")
    
    # Search for the task folder (could be in batch_1, batch_2, etc.)
    folder_path = None
    for item in os.listdir(rad_path):
        item_path = os.path.join(rad_path, item)
        if os.path.isdir(item_path):
            # Check if it's the task folder directly or in a batch subdirectory
            if os.path.basename(item_path) == config.name:
                folder_path = item_path
                break
            # Check in batch subdirectories
            elif item.startswith("batch_"):
                task_in_batch = os.path.join(item_path, config.name)
                if os.path.isdir(task_in_batch):
                    folder_path = task_in_batch
                    break
    
    assert folder_path is not None, f"Could not find task folder for {config.name} in text_only_tasks.rad"
    assert os.path.exists(folder_path), f"Folder path {folder_path} does not exist."

    with open(os.path.join(folder_path, "project_description.md"), "r") as f:
        project_description = f.read()

    with open(os.path.join(folder_path, "metadata.yaml"), "r") as f:
        metadata = AiraBenchTaskMetadata(**yaml.safe_load(f))

    prepare_code = os.path.join(folder_path, "prepare.py")
    evaluate_prepare_code = os.path.join(folder_path, "evaluate_prepare.py")
    evaluate_code = os.path.join(folder_path, "evaluate.py")

    # Only include Python files directly under folder_path, excluding the main scripts
    supporting_scripts = []
    for file in os.listdir(folder_path):
        if file.startswith("gold_submission"):  # skip gold_submission files
            continue
        if file.endswith(".py") and file not in [
            "prepare.py",
            "evaluate.py",
            "evaluate_prepare.py",
            "custom_labels.py",
            "download.py",
            "__init__.py",
        ]:
            supporting_scripts.append(os.path.join(folder_path, file))
    print(f"Found supporting scripts: {supporting_scripts}")

    return AiraBenchTaskData(
        project_description=project_description,
        prepare_code=prepare_code,
        metadata=metadata,
        evaluate_prepare_code=evaluate_prepare_code,
        evaluate_code=evaluate_code,
        supporting_scripts=supporting_scripts,
    )


class AiraBenchTask(Task):
    """
    Represents a simple task from AIRA Bench in which the agent is given
    a train dataset and a metric, and they are tasked with finding code that gets the best score
    on a holdout test dataset.

    Expected configuration keys (accessible via self.cfg):
      - dataset_code: The code to load the dataset. In self.prepare we call this and save it to disk.
      - project_description: The task description.
      - lower_is_better: Whether lower is better for the metric.
      - evaluate_file: A string representation of a python file that can be called to evaluate submission.
      ...
    """

    def __init__(
        self,
        cfg: AiraBenchTaskConfig,
    ) -> None:
        """
        Initialize the MLEBenchTask.

        Args:
            **cfg: Configuration containing keys like 'task_description' and 'answer'.
        """
        super().__init__(cfg)
        self.eval_every_step = cfg.eval_every_step
        self.task = get_aira_bench_task_from_config(cfg)

        # morph the various file locations into our standard terminology

        # extract dataset/config/splits from logging_info and ensure required keys are present
        log_info = self.task.metadata.logging_info
        required_keys = ("dataset", "config", "train_split", "test_split")
        missing_keys = [k for k in required_keys if k not in log_info]
        if missing_keys:
            raise RuntimeError(f"Missing required logging_info keys: {missing_keys}")
        dataset = log_info["dataset"]
        config_name = log_info["config"]
        train_split = log_info["train_split"]
        test_split = log_info["test_split"]

        def coorce_to_list(value):
            if isinstance(value, str):
                return [value]
            elif isinstance(value, list):
                assert all(isinstance(v, str) for v in value), "All elements of list must be strings"
                return value
            else:
                raise RuntimeError(f"Value must be a string or list of strings, got {type(value)}")

        all_config_names = coorce_to_list(config_name)
        all_train_splits = coorce_to_list(train_split)
        all_test_splits = coorce_to_list(test_split)

        # test the location of the first config and split
        config_name, train_split, test_split = all_config_names[0], all_train_splits[0], all_test_splits[0]
        primary_train = os.path.join(self.cfg.global_shared_data_dir, dataset, config_name, train_split)
        primary_test = os.path.join(self.cfg.global_shared_data_dir, dataset, config_name, test_split)
        if os.path.isdir(primary_train) and os.path.isdir(primary_test):
            self.global_shared_data_dir = self.cfg.global_shared_data_dir
            self.logger.info(f"✅ Using global_shared_data_dir: {self.global_shared_data_dir}")
        else:
            # check backup shared data dir
            backup_train = os.path.join(self.cfg.backup_global_shared_data_dir, dataset, config_name, train_split)
            backup_test = os.path.join(self.cfg.backup_global_shared_data_dir, dataset, config_name, test_split)
            if os.path.isdir(backup_train) and os.path.isdir(backup_test):
                self.global_shared_data_dir = self.cfg.backup_global_shared_data_dir
                self.logger.warning(
                    f"⚠️ Mitigated data not found; using backup global_shared_data_dir: {self.global_shared_data_dir}"
                )
            else:
                raise RuntimeError(
                    f"❌ Could not find data for {dataset}/{config_name} splits "
                    f"('{train_split}', '{test_split}') in either {self.cfg.global_shared_data_dir=} or {self.cfg.backup_global_shared_data_dir=}"
                )

        # check that all configs and splits exist in the same location as the first
        for config_name in all_config_names:
            for train_split in all_train_splits:
                for test_split in all_test_splits:
                    train_path = os.path.join(self.global_shared_data_dir, dataset, config_name, train_split)
                    test_path = os.path.join(self.global_shared_data_dir, dataset, config_name, test_split)
                    if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
                        raise RuntimeError(
                            f"❌ Could not find data for {dataset}/{config_name} splits "
                            f"('{train_split}', '{test_split}') in {self.global_shared_data_dir=}"
                        )

        # Making the data folders -- temporary to not clutter the /checkpoint/maui storage
        self.solver_data_mount = os.path.abspath(os.path.expanduser(self.cfg.data_dir))
        self.eval_data_mount = os.path.abspath(os.path.expanduser(self.cfg.eval_data_dir))

        # Use a dedicated agent log directory to avoid collisions where evaluate_prepare
        # may try to copy a submission file onto itself. Default to a `agent_logs`
        # subdirectory inside the solver data mount.
        self.agent_log_dir = os.path.join(self.solver_data_mount, "agent_logs")
        try:
            os.makedirs(self.agent_log_dir, exist_ok=True)
        except Exception:
            # If we can't create the directory (permission issues), fall back to solver_data_mount
            self.logger.warning(f"Could not create agent_log_dir={self.agent_log_dir}, falling back to solver_data_mount")
            self.agent_log_dir = self.solver_data_mount

        self.logger.info(
            f"""
            Task configured to use
            {self.global_shared_data_dir=}
            {self.solver_data_mount=}
            {self.eval_data_mount=}
            {self.agent_log_dir=}
            {self.eval_every_step=}
            {self.task.metadata.container_python_requirements=}
            {self.task.metadata.evaluate_container_python_requirements=}
            {self.task.metadata.file_export_globs=}
            {self.task.evaluate_prepare_code=}
            {self.task.evaluate_code=}
            {self.task.prepare_code=}
            {self.task.project_description=}
            {self.task.supporting_scripts=}
            """
        )

    def prepare(self, **task_args) -> Tuple[Dict, Dict]:
        state = task_args  # sneaky but this is where the interpreter is stored into state
        assert "solver_interpreter" in state
        assert os.path.samefile(state["solver_interpreter"].data_dir, self.solver_data_mount), (
            f"{state['solver_interpreter'].data_dir} != {self.solver_data_mount}"
        )
        assert "eval_interpreter" in state
        assert os.path.samefile(state["eval_interpreter"].data_dir, self.eval_data_mount), (
            f"{state['eval_interpreter'].data_dir} != {self.eval_data_mount}"
        )

        task_info = {
            TASK_DESCRIPTION: self.task.project_description,
            "lower_is_better": self.task.metadata.metric_lower_is_better,  # Important for solvers like Greedy
        }

        # grab the working dir of the interpreter, since this is where fetch_file pulls from
        self._interpreter_working_dir = state["solver_interpreter"].working_dir
        # will be used to pull files from the interpreter e.g
        # self._submission_file_path = os.path.join(self._interpreter_working_dir, "submission.csv")

        # Run prepare.py to load data into agent mount dir
        stdout, _, stderr = self.run_python_script(
            self.task.prepare_code,
            cmdline_args=[
                "--global-shared-data-dir",
                self.global_shared_data_dir,
                "--agent-data-mount-dir",
                self.solver_data_mount,
                "--agent-log-dir",
                self.agent_log_dir,
            ],
        )
        print(stdout)
        print(stderr)

        return state, task_info

    def step_task(self, state: Dict[str, Any], action: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a single step of the task. This is a no-op for this task.
        Submitting the answer is done via the 'solution' key in the evaluate fitness function.

        Args:
            state (Dict[str, Any]): The current state of the task.
            action (Any): The solution code (as a string) to evaluate.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the updated state and the outcome.
        """
        print("*" * 80, self.eval_every_step)
        self.logger.error(f"{'*' * 80} {self.eval_every_step}")
        state, outcome = self.exec_solution(state, action)
        print("*" * 80, self.eval_every_step)
        if self.eval_every_step:
            if not outcome[VALID_SOLUTION]:
                self.logger.error("Error during eval of test fitness: invalid solution")
                self.logger.debug("Skipping eval for this step...")
            print("*" * 80, "Running test evaluation!!!!!!")
            result, error = self.eval_state(state)
            if error:
                self.logger.error(f"Error during eval of test fitness: {error}")
                self.logger.debug("Skipping eval for this step...")
                outcome[TEST_FITNESS] = None
            else:
                self.logger.critical(f"Evalled during step task and got Test fitness: {result}")
                print("*" * 500, "RAN WITH A RESULT")
                outcome[TEST_FITNESS] = result

        return state, outcome

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # save solution to eval_data_mount as submitted_solution.py
        with open(os.path.join(self.eval_data_mount, "submitted_solution.py"), "w") as f:
            f.write(solution)

        state, outcome = self.exec_solution(state, solution)
        self.logger.info(f"exec_solution outcome: {outcome}")
        if outcome[VALID_SOLUTION] is False:
            outcome[TEST_FITNESS] = None
            return outcome

        result, error = self.eval_state(state)
        if error:
            self.logger.error(f"Error during eval: {error}")
            outcome[TEST_FITNESS] = None
        else:
            outcome[TEST_FITNESS] = result

        return outcome

    def close(self, state):
        pass

    def exec_solution(self, state, solution):
        """
        Runs the proposed solution code in the interpreter
        """
        # run the solution code
        # install container requirements
        reqs = [f'"{r}"' for r in self.task.metadata.container_python_requirements]
        install_command = "! pip install datasets==4.0.0 " + " ".join(reqs)
        interpreter: Interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(install_command, reset_session=True)
        print("\n".join(exec_output.term_out))

        datasets_check = """
import subprocess
result = subprocess.run(    "pip freeze | grep datasets",    shell=True,    capture_output=True,    text=True)
assert result.stdout.strip()=='datasets==4.0.0', f"Found: {result.stdout.strip()}"
"""
        solution = datasets_check + "\n" + solution
        interpreter: Interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(
            solution, reset_session=False
        )  # thomfoster NOTE: greedy.py passes in a filename here too
        outcome = {EXECUTION_OUTPUT: exec_output, VALID_SOLUTION: True}

        # thomfoster NOTE: mlebench/task.py checks if execution was successful
        # and removes any submission.csv file produced if unsuccessful

        return state, outcome

    def eval_state(self, state):
        """
        Try to fetch the submission file from the solver interpreter
        and run the evaluate script on it.
        Returns metrics, error
        where error is not None iff there was an error
        """

        # export the globs into the log dir
        solver_interpreter: Interpreter = state["solver_interpreter"]
        for file_glob in self.task.metadata.file_export_globs:
            fetch_path = os.path.join(self._interpreter_working_dir, file_glob)
            was_fetched = solver_interpreter.fetch_file(fetch_path)
            if not was_fetched:
                return None, f"Failed to fetch {file_glob}"
            shutil.copy(fetch_path, self.agent_log_dir)

        # run evaluate_prepare.py
        self.run_python_script(
            self.task.evaluate_prepare_code,
            cmdline_args=[
                "--global-shared-data-dir",
                self.global_shared_data_dir,
                "--agent-data-mount-dir",
                self.eval_data_mount,
                "--agent-log-dir",
                self.agent_log_dir,
            ],
        )

        # write self.task.evaluate_code to a file in --agent-data-mount-dir
        with open(self.task.evaluate_code, "r") as f:
            evaluate_code = f.read()
        with open(os.path.join(self.eval_data_mount, "evaluate.py"), "w") as f:
            f.write(evaluate_code)

        # write any supporting scripts to the eval_data_mount dir
        if self.task.supporting_scripts:
            for script_path in self.task.supporting_scripts:
                with open(script_path, "r") as f:
                    script_code = f.read()
                script_name = os.path.basename(script_path)
                with open(os.path.join(self.eval_data_mount, script_name), "w") as f:
                    f.write(script_code)

        # Install the evaluate contaner requirements
        reqs = [f'"{r}"' for r in self.task.metadata.evaluate_container_python_requirements]
        install_command = "! pip install datasets==4.0.0 " + " ".join(reqs)
        interpreter: Interpreter = state["eval_interpreter"]
        exec_output: ExecutionResult = interpreter.run(install_command, reset_session=True)

        # run evalaute.py inside the container (as you did with exec_solution)
        run_evaluate_command = "! python ./data/evaluate.py --submission-file ./data/submission.csv"
        eval_interpreter: Interpreter = state["eval_interpreter"]
        datasets_check = """
import subprocess
result = subprocess.run(    "pip freeze | grep datasets",    shell=True,    capture_output=True,    text=True)
assert result.stdout.strip()=='datasets==4.0.0', f"Found: {result.stdout.strip()}"
"""
        run_evaluate_command = datasets_check + "\n" + run_evaluate_command
        exec_output: ExecutionResult = eval_interpreter.run(run_evaluate_command, reset_session=False)
        term_out_lines = exec_output.term_out
        stdout = "\n".join(term_out_lines)

        # extract the results from the stdout
        pattern = re.compile(r"--- EVALUATION RESULT ---\s*(\{[\s\S]*?\})", re.DOTALL)
        m = pattern.search(stdout)
        if not m:
            return None, f"Failed to extract results dict string from stdout: `{stdout}`"

        result_dict_str = m.group(1)
        try:
            result_dict = json.loads(result_dict_str)
        except:  # noqa: E722
            return None, f"Failed to turn results dict string into dict: `{result_dict_str}`"

        if len(result_dict) == 0:
            return None, f"Empty results dict: `{result_dict_str}`"
        elif len(result_dict) == 1:
            result = list(result_dict.values())[0]
            return result, None
        else:
            self.logger.warning(f"Multiple results found in results dict: `{result_dict}`")
            return result_dict, None

    def run_python_script(self, fpath, cmdline_args=None):
        """
        Runs the python script at fpath, capturing stdout and its globals.
        Supports sibling imports like `import sibling`.
        """
        if cmdline_args is None:
            cmdline_args = []

        script_dir = os.path.abspath(os.path.dirname(fpath))

        old_argv = sys.argv[:]
        old_sys_path = sys.path[:]
        old_cwd = os.getcwd()

        try:
            # Emulate `python file.py`
            sys.argv = [fpath, *cmdline_args]
            sys.path.insert(0, script_dir)  # make sibling imports work
            os.chdir(script_dir)  # optional but helps with relative file I/O

            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                ns = runpy.run_path(fpath, run_name="__main__")

            stdout = buf_out.getvalue()
            stderr = buf_err.getvalue()
        finally:
            sys.argv = old_argv
            sys.path = old_sys_path
            os.chdir(old_cwd)

        return stdout, ns, stderr