# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.response import wrap_code
from dojo.core.solvers.utils.journal import Node

analyze_schema_without_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings."
        }
    },
    "required": ["is_bug", "summary"]
}"""

analyze_schema_with_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings. DO NOT suggest fixes or improvements."
        },
        "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
            }
    },
    "required": ["is_bug", "summary", "metric"]
}"""


def analyze_op(
    analyze_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    input_node: Node,
    fetch_metric: bool = True,
) -> str:
    code = input_node.code
    execution_output = input_node.term_out

    analyze_data = {
        "task_desc": task_description,
        "code": wrap_code(code),
        "execution_output": wrap_code(execution_output, lang=""),
    }

    schema = analyze_schema_with_eval if fetch_metric else analyze_schema_without_eval

    return analyze_llm(
        query_data=analyze_data,
        json_schema=schema,
        function_name="submit_review",
        function_description="Submit a review evaluating the output of the training script.",
        no_user_message=True,
    )
