# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

from dojo.solvers.mcts.mcts import MCTS, MCTSNode, normalise_q_value
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.config_dataclasses.solver.sa_ucb import SA_UCBSolverConfig
from dojo.utils.code_parsing import parse_json_output


class SA_UCBNode(MCTSNode):
    """Extended MCTS node with semantic scoring for promise and novelty."""

    promise_score: Optional[float] = None
    novelty_score: Optional[float] = None
    promise_reasoning: str = ""
    novelty_reasoning: str = ""

    def extra_metrics_to_log(self):
        base_metrics = super().extra_metrics_to_log()
        base_metrics.update({
            "promise_score": self.promise_score,
            "novelty_score": self.novelty_score,
            "promise_reasoning": self.promise_reasoning,
            "novelty_reasoning": self.novelty_reasoning,
        })
        return base_metrics


def sa_uct_value(
    q_value: float,
    explore_count: int,
    parent_explore_count: int,
    uct_c: float,
    global_max_q_val: float,
    global_min_q_val: float,
    promise_score: Optional[float],
    novelty_score: Optional[float],
    lambda_promise: float,
    gamma_novelty: float,
):
    """
    Calculate the SA-UCB (Semantic Aware UCB) value for a node.

    SA-UCB(v) = (Q_v / N_v) + C * sqrt(ln(N_parent) / N_v) + λ * P_v + γ * U_v

    Args:
        q_value: Node's Q-value (cumulative reward / visit count)
        explore_count: Number of times this node was visited
        parent_explore_count: Number of times parent node was visited
        uct_c: UCT exploration constant
        global_max_q_val: Global maximum Q-value for normalization
        global_min_q_val: Global minimum Q-value for normalization
        promise_score: Promise score P_v from promise scorer LLM (0-1 range)
        novelty_score: Novelty score U_v from novelty scorer LLM (0-1 range)
        lambda_promise: Weight for promise bonus
        gamma_novelty: Weight for novelty bonus

    Returns:
        SA-UCB value for node selection
    """
    if explore_count == 0:
        return -1e8

    # Standard UCB component
    norm_q = normalise_q_value(q_value, global_max_q_val, global_min_q_val)
    exploration = math.sqrt(math.log(parent_explore_count) / explore_count)
    standard_ucb = norm_q + uct_c * exploration

    # Semantic bonus component
    promise_bonus = lambda_promise * (promise_score if promise_score is not None else 0.0)
    novelty_bonus = gamma_novelty * (novelty_score if novelty_score is not None else 0.0)
    semantic_bonus = promise_bonus + novelty_bonus

    return standard_ucb + semantic_bonus


class SA_UCB(MCTS):
    """
    Semantic Aware UCB (SA-UCB) solver extending MCTS.

    This solver enhances standard MCTS with semantic awareness by using two
    specialized LLMs to score nodes:
    - Promise Scorer: Evaluates how promising a solution approach appears
    - Novelty Scorer: Evaluates how novel/diverse a solution is vs siblings

    The SA-UCB formula is:
    SA-UCB(v) = Standard_UCB(v) + λ * P_v + γ * U_v

    where P_v is the promise score and U_v is the novelty score.
    """

    def __init__(self, cfg: SA_UCBSolverConfig, task_info):
        """
        Initialize the SA-UCB solver.

        Args:
            cfg: SA-UCB solver configuration
            task_info: Dictionary containing task information
        """
        super().__init__(cfg, task_info)

        # SA-UCB specific hyperparameters
        self.lambda_promise = cfg.lambda_promise
        self.gamma_novelty = cfg.gamma_novelty

        # Create logging directory for scorer outputs
        self.scorer_log_dir = Path(cfg.checkpoint_path) / "scorer_logs"
        self.scorer_log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"SA-UCB initialized with λ={self.lambda_promise}, γ={self.gamma_novelty}")
        self.logger.info(f"Scorer logs will be saved to: {self.scorer_log_dir}")

    def setup_operators(self):
        """
        Initialize and configure the LLM operators including semantic scorers.

        Extends the base MCTS setup_operators to add promise and novelty scorers.
        """
        # Call parent setup for standard operators
        super().setup_operators()

        # Set up the semantic scoring LLMs
        self.promise_scorer_llm = GenericLLM(self.cfg.operators["promise_scorer"])
        self.novelty_scorer_llm = GenericLLM(self.cfg.operators["novelty_scorer"])

        self.logger.info("Promise and Novelty scorer LLMs initialized")

    def create_root_node(self):
        """Create root node using SA_UCBNode instead of MCTSNode."""
        from dojo.core.solvers.utils.metric import WorstMetricValue

        self.root_node = SA_UCBNode(
            code="",
            plan="",
            analysis="",
            metric=WorstMetricValue(maximize=not self.lower_is_better),
            is_buggy=True,
        )
        self.root_node.absorb_exec_result(None)
        self.journal.append(self.root_node)
        self.log_journal()
        self.state.current_step += 1

    def search_policy(self, root_node: SA_UCBNode) -> List[SA_UCBNode]:
        """
        Traverse the tree from root to leaf using SA-UCT selection.

        This overrides the base MCTS search_policy to use SA-UCB values
        instead of standard UCB values.

        Args:
            root_node: Starting node for path selection

        Returns:
            List of nodes representing path from root to selected leaf
        """
        path: List[SA_UCBNode] = []
        current_node = root_node

        while True:
            path.append(current_node)
            if not current_node.children:
                # It's a leaf
                return path

            current_node = max(
                current_node.children,
                key=lambda c: sa_uct_value(
                    q_value=c.q_value(self.lower_is_better),
                    explore_count=c.explore_count,
                    parent_explore_count=current_node.explore_count,
                    uct_c=self.cfg.uct_c,
                    global_max_q_val=self.global_max_q_val,
                    global_min_q_val=self.global_min_q_val,
                    promise_score=c.promise_score,
                    novelty_score=c.novelty_score,
                    lambda_promise=self.lambda_promise,
                    gamma_novelty=self.gamma_novelty,
                ),
            )

    def _score_promise(self, node: SA_UCBNode) -> Tuple[float, str]:
        """
        Score how promising a solution approach appears.

        Args:
            node: Node to score for promise

        Returns:
            Tuple of (promise_score, reasoning)
        """
        query_data = {
            "task_desc": self.task_desc,
            "plan": node.plan,
            "code": node.code[:2000],  # Truncate to avoid context limits
            "parent_plan": node.parents[0].plan if node.parents else "",
        }

        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "promise_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Promise score between 0 and 1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the promise score"
                }
            },
            "required": ["promise_score", "reasoning"]
        }

        try:
            response, metrics = self.promise_scorer_llm(
                query_data=query_data,
                json_schema=json.dumps(json_schema),
                function_name="score_promise",
                function_description="Score the promise of a solution approach"
            )

            # Parse the response
            parsed = parse_json_output(response)

            promise_score = float(parsed.get("promise_score", 0.5))
            reasoning = parsed.get("reasoning", "No reasoning provided")

            # Clamp to [0, 1] range
            promise_score = max(0.0, min(1.0, promise_score))

            # Log the scorer output
            self._log_scorer_output(
                node_step=node.step,
                scorer_type="promise",
                score=promise_score,
                reasoning=reasoning,
                metrics=metrics
            )

            self.logger.info(f"Promise score for node {node.step}: {promise_score:.3f}")

            return promise_score, reasoning

        except Exception as e:
            self.logger.error(f"Error scoring promise: {str(e)}")
            return 0.5, f"Error during scoring: {str(e)}"

    def _score_novelty(self, node: SA_UCBNode, siblings: List[SA_UCBNode]) -> Tuple[float, str]:
        """
        Score how novel/diverse a solution is compared to its siblings.

        Args:
            node: Node to score for novelty
            siblings: List of sibling nodes for comparison

        Returns:
            Tuple of (novelty_score, reasoning)
        """
        # Prepare sibling summaries
        sibling_summaries = []
        for i, sibling in enumerate(siblings[:5]):  # Limit to 5 siblings to avoid context overflow
            if sibling.step != node.step:  # Don't include the node itself
                sibling_summaries.append({
                    "index": i,
                    "plan": sibling.plan[:300],  # Truncate
                })

        query_data = {
            "task_desc": self.task_desc,
            "plan": node.plan,
            "code": node.code[:2000],  # Truncate
            "sibling_plans": json.dumps(sibling_summaries, indent=2),
            "num_siblings": len(siblings) - 1,  # Exclude self
        }

        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "novelty_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Novelty score between 0 and 1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the novelty score"
                }
            },
            "required": ["novelty_score", "reasoning"]
        }

        try:
            response, metrics = self.novelty_scorer_llm(
                query_data=query_data,
                json_schema=json.dumps(json_schema),
                function_name="score_novelty",
                function_description="Score the novelty of a solution compared to siblings"
            )

            # Parse the response
            parsed = parse_json_output(response)

            novelty_score = float(parsed.get("novelty_score", 0.5))
            reasoning = parsed.get("reasoning", "No reasoning provided")

            # Clamp to [0, 1] range
            novelty_score = max(0.0, min(1.0, novelty_score))

            # Log the scorer output
            self._log_scorer_output(
                node_step=node.step,
                scorer_type="novelty",
                score=novelty_score,
                reasoning=reasoning,
                metrics=metrics
            )

            self.logger.info(f"Novelty score for node {node.step}: {novelty_score:.3f}")

            return novelty_score, reasoning

        except Exception as e:
            self.logger.error(f"Error scoring novelty: {str(e)}")
            return 0.5, f"Error during scoring: {str(e)}"

    def _log_scorer_output(
        self,
        node_step: int,
        scorer_type: str,
        score: float,
        reasoning: str,
        metrics: Dict[str, Any]
    ):
        """
        Log scorer LLM output to file for analysis.

        Args:
            node_step: Step number of the node being scored
            scorer_type: Type of scorer ('promise' or 'novelty')
            score: The computed score
            reasoning: The reasoning provided by the LLM
            metrics: Metrics from the LLM call (usage stats, prompts, etc.)
        """
        log_entry = {
            "timestamp": time.time(),
            "node_step": node_step,
            "scorer_type": scorer_type,
            "score": score,
            "reasoning": reasoning,
            "system_prompt": None,
            "user_prompt": None,
            "completion": None,
            "usage": metrics.get("usage", {}),
        }

        # Extract prompts if available
        if "prompt_messages" in metrics:
            messages = metrics["prompt_messages"]
            for msg in messages:
                if msg.get("role") == "system":
                    log_entry["system_prompt"] = msg.get("content", "")
                elif msg.get("role") == "user":
                    log_entry["user_prompt"] = msg.get("content", "")

        if "completion_text" in metrics:
            log_entry["completion"] = metrics["completion_text"]

        # Write to JSONL file
        log_file = self.scorer_log_dir / f"{scorer_type}_scores.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also log to main logger
        self.logger.log(
            {
                f"{scorer_type}_score": score,
                f"{scorer_type}_reasoning": reasoning,
                "node_step": node_step,
            },
            f"SCORER_{scorer_type.upper()}",
            step=self.state.current_step,
        )

    def _draft(self, parent: Optional[SA_UCBNode] = None) -> SA_UCBNode:
        """
        Generate a new solution from scratch using the draft LLM operator.

        Extends base _draft to use SA_UCBNode and score the new node.

        Returns:
            SA_UCBNode: A new node containing the drafted solution with scores
        """
        from dojo.core.solvers.operators.core import execute_op_plan_code
        from dojo.solvers.utils import get_complextiy_level

        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            get_complextiy_level(parent) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )

        node = SA_UCBNode(
            plan=plan,
            code=code,
            parents=[parent],
            operators_used=["draft"],
            operators_metrics=[metrics]
        )

        self.logger.info(f"Draft Node Created - Metrics: {metrics}")

        # Score the newly drafted node
        if parent is not None:
            # Get siblings (children of the same parent)
            siblings = list(parent.children) if parent.children else []
            siblings.append(node)  # Include the new node

            # Score promise and novelty
            node.promise_score, node.promise_reasoning = self._score_promise(node)
            node.novelty_score, node.novelty_reasoning = self._score_novelty(node, siblings)
        else:
            # Root node - no scoring needed
            node.promise_score = 0.5
            node.novelty_score = 0.5
            node.promise_reasoning = "Root node - no scoring"
            node.novelty_reasoning = "Root node - no scoring"

        return node

    def _improve(self, parent_node: SA_UCBNode) -> SA_UCBNode:
        """
        Improve an existing solution using the improve LLM operator.

        Extends base _improve to use SA_UCBNode and score the new node.

        Args:
            parent_node: The node containing the solution to improve

        Returns:
            SA_UCBNode: A new node containing the improved solution with scores
        """
        from dojo.core.solvers.operators.core import execute_op_plan_code
        from dojo.solvers.utils import get_complextiy_level

        plan, code, metrics = execute_op_plan_code(
            self.improve_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            get_complextiy_level(parent_node) if self.cfg.use_complexity else None,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )

        node = SA_UCBNode(
            plan=plan,
            code=code,
            parents=[parent_node],
            operators_used=["improve"],
            operators_metrics=[metrics]
        )

        self.logger.info(f"Improve Node Created - Metrics: {metrics}")

        # Get siblings for novelty scoring
        siblings = list(parent_node.children) if parent_node.children else []
        siblings.append(node)  # Include the new node

        # Score promise and novelty
        node.promise_score, node.promise_reasoning = self._score_promise(node)
        node.novelty_score, node.novelty_reasoning = self._score_novelty(node, siblings)

        return node

    def _debug(self, parent_node: SA_UCBNode) -> SA_UCBNode:
        """
        Debug a buggy solution using the debug LLM operator.

        Extends base _debug to use SA_UCBNode and score the new node.

        Args:
            parent_node: The node containing the buggy solution to debug

        Returns:
            SA_UCBNode: A new node containing the debugged solution with scores
        """
        from dojo.core.solvers.operators.core import execute_op_plan_code

        plan, code, metrics = execute_op_plan_code(
            self.debug_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )

        node = SA_UCBNode(
            plan=plan,
            code=code,
            parents=[parent_node],
            operators_used=["debug"],
            operators_metrics=[metrics]
        )

        self.logger.info(f"Debug Node Created - Metrics: {metrics}")

        # Get siblings for novelty scoring
        siblings = list(parent_node.children) if parent_node.children else []
        siblings.append(node)  # Include the new node

        # Score promise and novelty
        node.promise_score, node.promise_reasoning = self._score_promise(node)
        node.novelty_score, node.novelty_reasoning = self._score_novelty(node, siblings)

        return node
