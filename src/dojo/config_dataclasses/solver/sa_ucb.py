# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import MISSING

from dojo.config_dataclasses.solver.mcts import MCTSSolverConfig


@dataclass
class SA_UCBSolverConfig(MCTSSolverConfig):
    """
    Configuration for the Semantic Aware UCB (SA-UCB) solver.

    Extends MCTSSolverConfig with additional hyperparameters for semantic bonuses.
    """

    # --- Semantic Scoring Configuration ---
    lambda_promise: float = field(
        default=MISSING,
        metadata={
            "description": "Weight for promise score bonus in SA-UCB formula. "
            "Controls how much the promise score influences node selection. "
            "Recommended range: 0.1 - 0.5"
        }
    )

    gamma_novelty: float = field(
        default=MISSING,
        metadata={
            "description": "Weight for novelty score bonus in SA-UCB formula. "
            "Controls how much diversity/novelty influences node selection. "
            "Recommended range: 0.1 - 0.5"
        }
    )

    def validate(self) -> None:
        super().validate()

        # Validate semantic scoring hyperparameters
        if self.lambda_promise < 0:
            raise ValueError(f"lambda_promise must be non-negative, got {self.lambda_promise}")

        if self.gamma_novelty < 0:
            raise ValueError(f"gamma_novelty must be non-negative, got {self.gamma_novelty}")
