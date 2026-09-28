"""Adapter for the ``deeppr`` package using the shared ``phaseretrieval`` package.

Identical canonical interface to ``dpr_legacy.py``. The phase retrieval algorithms come from
the PhaseRetrieval repository, located by ``REG_PR_ROOT`` (default: ``../PhaseRetrieval``
next to the DPR checkout).
"""
import os
import sys

import numpy as np
import torch

from adapters.dpr_legacy import CDT, F64, FDT, Adapter as _LegacyAdapter

NAME = "dpr_modern"


class Adapter(_LegacyAdapter):
    name = NAME
    supports_toggle = True

    def __init__(self, code_root):
        self.root = os.path.abspath(code_root)
        pr_root = os.path.abspath(os.environ.get("REG_PR_ROOT", os.path.join(self.root, "..", "PhaseRetrieval")))
        sys.path.insert(0, pr_root)
        sys.path.insert(0, self.root)
        torch.set_num_threads(1)
        if F64:
            torch.set_default_dtype(torch.float64)
        import deeppr
        from deeppr import weightedpartialconv2d
        from phaseretrieval import PhaseRetrieval

        self.dpr = deeppr
        self.WeightedPartialConv2d = weightedpartialconv2d.WeightedPartialConv2d
        self.PhaseRetrieval = PhaseRetrieval
        self.notes = list(self.notes) + ["phaseretrieval imported from {}".format(pr_root)]
        self._models = {}
        self._demo = None

    def phase_retrieval(self, amplitude, support, unknown, info, iteration, initial_phase, toggle=False):
        t = lambda x: torch.from_numpy(np.asarray(x, dtype=FDT))[None, None]
        it = self.PhaseRetrieval(t(amplitude), t(support), t(unknown), **dict(info))
        phase = torch.from_numpy(np.asarray(initial_phase, dtype=CDT))[:, None]
        with torch.no_grad():
            out, path = it(iteration, phase, toggle=toggle, **dict(info))
        return out[:, 0].numpy(), path.numpy()
