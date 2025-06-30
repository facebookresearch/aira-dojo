# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import Callable, Optional

import humanize
from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import wrap_code
from dojo.solvers.utils import Complexity


def crossover_op(
    crossover_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    input_node1: Node,
    input_node2: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    pkgs = cfg.available_packages
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    steps_remaining = cfg.step_limit - step_count

    improve_data = {
        "task_desc": task_description,
        "prev_code1": wrap_code(input_node1.code),
        "prev_terminal_output1": wrap_code(input_node1.term_out, lang=""),
        "prev_code2": wrap_code(input_node2.code),
        "prev_terminal_output2": wrap_code(input_node2.term_out, lang=""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "improve_complexity": None,
        "memory": None,
        "data_overview": None,
        "packages": pkg_str,
    }

    if cfg.data_preview and data_preview is not None:
        improve_data["data_overview"] = data_preview
    else:
        improve_data["data_overview"] = "(No data preview available)"

    return crossover_llm(query_data=improve_data, no_user_message=True)
