"""Adapter for the ``deeppr`` package using the shared ``phaseretrieval`` package.

Identical canonical interface to ``dpr_legacy.py``. The phase retrieval algorithms, including
``find_center`` and ``align_object``, come from the PhaseRetrieval repository, located by
``REG_PR_ROOT`` (default: ``../PhaseRetrieval`` next to the DPR checkout). The helper functions
of ``demo.ipynb`` (prepare_data, refine_by_pr) are executed from the notebook's own code cells:
the cells that contain only imports and function definitions.
"""

import ast
import json
import os
import sys

import numpy as np
import torch

from adapters.dpr_legacy import CDT, F64, FDT
from adapters.dpr_legacy import Adapter as _LegacyAdapter

NAME = "dpr_modern"


class Adapter(_LegacyAdapter):
    name = NAME
    supports_toggle = True

    def __init__(self, code_root):
        self.root = os.path.abspath(code_root)
        pr_root = os.path.abspath(
            os.environ.get("REG_PR_ROOT", os.path.join(self.root, "..", "PhaseRetrieval"))
        )
        sys.path.insert(0, pr_root)
        sys.path.insert(0, self.root)
        torch.set_num_threads(1)
        if F64:
            torch.set_default_dtype(torch.float64)
        from phaseretrieval import PhaseRetrieval, align_object, find_center

        import deeppr
        from deeppr import weightedpartialconv2d

        self.dpr = deeppr
        self.WeightedPartialConv2d = weightedpartialconv2d.WeightedPartialConv2d
        self.PhaseRetrieval = PhaseRetrieval
        self.align_object = align_object
        self.find_center = find_center
        self.notes = list(self.notes) + [f"phaseretrieval imported from {pr_root}"]
        self._models = {}
        self._demo = None

    def phase_retrieval(
        self, amplitude, support, unknown, info, iteration, initial_phase, toggle=False
    ):
        def t(x):
            return torch.from_numpy(np.asarray(x, dtype=FDT))[None, None]

        it = self.PhaseRetrieval(t(amplitude), t(support), t(unknown), **dict(info))
        phase = torch.from_numpy(np.asarray(initial_phase, dtype=CDT))[:, None]
        with torch.no_grad():
            out, path = it(iteration, phase, toggle=toggle, **dict(info))
        return out[:, 0].numpy(), path.numpy()

    # ---- loss.py: CombinedLoss aligns with phaseretrieval.align_object ---------------------
    def loss_align_obj(self, output, target, limit=32):
        assert limit == max(output.shape[-2:]) // 2, "align_object uses limit = max(H, W) // 2"
        return self.align_object(torch.from_numpy(output), torch.from_numpy(target)).numpy()

    # ---- demo.ipynb pipeline ------------------------------------------------------------------
    def demo(self):
        if self._demo is None:
            with open(os.path.join(self.root, "demo.ipynb")) as f:
                nb = json.load(f)
            ns = {}
            for cell in nb["cells"]:
                source = "".join(cell["source"])
                if cell["cell_type"] == "code" and all(
                    isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                    for node in ast.parse(source).body
                ):
                    exec(source, ns)
            self._demo = ns
        return self._demo

    def prepare_data(self, pattern):
        return self.demo()["prepare_data"](np.array(pattern))

    def demo_pipeline(self, pattern, ckpt, param, n_iter):
        ns = self.demo()
        inp, mask = self.prepare_data(pattern)
        with torch.no_grad():
            x = inp * mask
            dpr = self.align_object(self.model(ckpt)(x, mask))
            refined, error = ns["refine_by_pr"](
                x, mask, dpr, n_iter, dict(param), torch.device("cpu"), return_error=True
            )
            refined = self.align_object(refined, dpr)
        return {
            "input": inp.numpy(),
            "mask": mask.numpy(),
            "dpr": dpr.numpy(),
            "dpr_refined": refined.numpy(),
            "refine_error": error.numpy(),
            "center_shift": np.asarray(self.find_center(np.array(pattern))),
        }
