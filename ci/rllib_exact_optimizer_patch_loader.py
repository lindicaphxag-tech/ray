"""Load EXACT patched TorchLearner._set_optimizer_state into installed Ray.

The fork source tree is newer than the PyPI Ray 2.59 wheel. Loading the
*whole* source file would mix incompatible API versions; this isolates the one
changed method from the real source AST and installs it on native wheels.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import ray.rllib.core.learner.torch.torch_learner as installed
from ray.rllib.utils.torch_utils import copy_torch_tensors

SOURCE=Path("/tmp/patched_torch_learner.py")


def get_source_exact_method():
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"),filename=str(SOURCE))
    klass=next(node for node in tree.body
               if isinstance(node,ast.ClassDef) and node.name=="TorchLearner")
    function=copy.deepcopy(next(node for node in klass.body
               if isinstance(node,ast.FunctionDef)
               and node.name=="_set_optimizer_state"))
    function.decorator_list=[]
    function.returns=None
    for arg in [*function.args.args,*function.args.kwonlyargs]:
        arg.annotation=None
    scope=dict(vars(installed))
    scope["copy_torch_tensors"]=copy_torch_tensors
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[function],type_ignores=[])),str(SOURCE),"exec"),scope)
    return scope["_set_optimizer_state"]


def patch_class():
    installed.TorchLearner._set_optimizer_state=get_source_exact_method()
    return True


def patch_instance(learner):
    learner._set_optimizer_state=get_source_exact_method().__get__(learner,type(learner))
    return True
