# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import II, SI

from dojo.config_dataclasses.task.base import TaskConfig


@dataclass
class AiraBenchTaskConfig(TaskConfig):
    eval_every_step: bool = field(
        default=True,
        metadata={
            "help": "Whether to evaluate the solution everytime .step_task is called or not.",
        },
    )

    data_dir: str = field(
        default=II("get_tmp_dir:train_data"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )

    requires_eval_interpreter: bool = field(
        default=True,
        metadata={
            "help": "Whether the task requires a separate interpreter for evaluation.",
        },
    )
    eval_data_dir: str = field(
        default=II("get_tmp_dir:eval_data"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )

    # This should be set by vendor
    global_shared_data_dir: str = field(
        default="/aira-bench-data/datasets/airs_text_vendor",
        metadata={
            "help": "The directory containing all predownloaded datasets, that aira-bench-data task folders use in prepare.py to populate data_dir and eval_data_dir",
            "exclude_from_hash": True,
        },
    )

    # This should be set by vendor
    backup_global_shared_data_dir: str = field(
        default="/aira-bench-data/datasets/airs_text_vendor_backup",
        metadata={
            "help": "A backup directory containing all predownloaded datasets, that aira-bench-data task folders use in prepare.py to populate data_dir and eval_data_dir if the primary global_shared_data_dir is not available",
            "exclude_from_hash": True,
        },
    )
    interpreter_working_dir: str = field(
        default=SI("${interpreter.working_dir}"),
        metadata={
            "help": "The directory on the node that is mounted to the agents home dir. This is the place where the interpreter has write access. This serves for solutions to store the submission files (`.csv`s), and python code to be executed (solution.py).",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()