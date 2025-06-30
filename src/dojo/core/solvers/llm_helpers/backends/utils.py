# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient
from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from dojo.core.solvers.llm_helpers.backends.open_ai import OpenAIClient


def get_client(client_cfg):
    match client_cfg.api:
        case "openai":
            return OpenAIClient(client_cfg)
        case "litellm":
            return LiteLLMClient(client_cfg)
        case "gdm":
            return GDMClient(client_cfg)
        case _:
            raise Exception(f"Unknown API: {client_cfg['api']}")
