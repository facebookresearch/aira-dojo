# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dojo.core.interpreters.python import PythonInterpreter
from dojo.core.interpreters.jupyter.jupyter_interpreter import JupyterInterpreterFactory

INTERPRETER_MAP = {"PythonInterpreterConfig": PythonInterpreter, "JupyterInterpreterConfig": JupyterInterpreterFactory}
