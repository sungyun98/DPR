"""Adapter for the original DPR code (tag v1.0-legacy, package ``module``, PyTorch 2.1).

Canonical NumPy interface used by the regression cases:

* real images ``(N, H, W)`` float32, complex images ``(N, H, W)`` complex64
* network inputs ``(1, 1, 512, 512)`` float32 with a boolean mask of the same shape

The helper functions of ``demo.ipynb`` (find_center, prepare_data, align_obj_cen, align_obj,
refine_by_pr) are executed from the notebook's own code cell so that the demo pipeline is
tested exactly as written.
"""
import json
import os
import sys

import numpy as np
import torch

# float64 diagnostic mode for the phase retrieval cases (REG_FLOAT64=1)
F64 = os.environ.get("REG_FLOAT64") == "1"
FDT, CDT = (np.float64, np.complex128) if F64 else (np.float32, np.complex64)

NAME = "dpr_legacy"
NOTES = ["torch.set_num_threads(1) for bit-reproducible CPU results",
         "checkpoints loaded with map_location='cpu' (as demo.ipynb does with its device)"]


class Adapter:
    name = NAME
    notes = NOTES
    supports_toggle = False  # DPR's PhaseRetrieval.forward has no toggle (k-space) output

    def __init__(self, code_root):
        self.root = os.path.abspath(code_root)
        sys.path.insert(0, self.root)
        torch.set_num_threads(1)
        if F64:
            torch.set_default_dtype(torch.float64)
        import module as dpr  # noqa: F401  (the original package name)
        from module import weightedpartialconv2d

        self.dpr = dpr
        self.WeightedPartialConv2d = weightedpartialconv2d.WeightedPartialConv2d
        self._models = {}
        self._demo = None

    # ---- centred FFT helpers ---------------------------------------------------------
    def fft2c(self, x):
        return self.dpr._fft2(torch.from_numpy(x)).numpy()

    def ifft2c(self, x):
        return self.dpr._ifft2(torch.from_numpy(x)).numpy()

    # ---- phase retrieval (module/phaseretrieval.py) ------------------------------------
    def phase_retrieval(self, amplitude, support, unknown, info, iteration, initial_phase, toggle=False):
        if toggle:
            raise NotImplementedError("toggle is not supported by the DPR implementation")
        t = lambda x: torch.from_numpy(np.asarray(x, dtype=FDT))[None, None]
        it = self.dpr.PhaseRetrieval(t(amplitude), t(support), t(unknown), **dict(info))
        phase = torch.from_numpy(np.asarray(initial_phase, dtype=CDT))[:, None]
        with torch.no_grad():
            out, path = it(iteration, phase, **dict(info))
        return out[:, 0].numpy(), path.numpy()

    # ---- network ------------------------------------------------------------------------
    def model(self, ckpt):
        if ckpt not in self._models:
            m = self.dpr.Network(ngf=64, max_features=1024, weight_model=True, downsample_FFC=False, refinement=True)
            path = os.path.join(self.root, "pretrained", ckpt)
            m.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
            self._models[ckpt] = m.eval()
        return self._models[ckpt]

    def network(self, ckpt, inp, mask, false_scale=False):
        x, m = torch.from_numpy(inp), torch.from_numpy(mask)
        with torch.no_grad():
            return self.model(ckpt)(x * m, m, false_scale=false_scale).numpy()

    # ---- demo.ipynb pipeline -------------------------------------------------------------------
    def demo(self):
        if self._demo is None:
            with open(os.path.join(self.root, "demo.ipynb")) as f:
                nb = json.load(f)
            code = [c for c in nb["cells"] if c["cell_type"] == "code"]
            ns = {}
            exec("".join(code[0]["source"]), ns)  # imports (module import resolved via sys.path)
            exec("".join(code[1]["source"]), ns)  # helper functions
            self._demo = ns
        return self._demo

    def demo_pipeline(self, pattern, ckpt, param, n_iter):
        ns = self.demo()
        inp, mask = ns["prepare_data"](np.array(pattern), bin=None)
        with torch.no_grad():
            x = inp * mask
            dpr = ns["align_obj_cen"](self.model(ckpt)(x, mask))
            refined, error = ns["refine_by_pr"](x, mask, dpr, n_iter, dict(param), torch.device("cpu"), err_out=True)
            refined = ns["align_obj"](refined, dpr.clone())
        return {"input": inp.numpy(), "mask": mask.numpy(), "dpr": dpr.numpy(), "dpr_refined": refined.numpy(),
                "refine_error": error.numpy(), "center_shift": np.asarray(ns["find_center"](np.array(pattern)))}

    # ---- dataset.py ----------------------------------------------------------------------------
    def generate_diffraction(self, obj, seed, **kwargs):
        torch.manual_seed(seed)
        inten, target = self.dpr.GenerateDiffraction(torch.from_numpy(obj), device="cpu", **kwargs)
        return inten.numpy(), target.numpy()

    def binarize(self, image_u8, threshold):
        from PIL import Image
        return np.asarray(self.dpr.Binarize(threshold)(Image.fromarray(image_u8)))

    def dilate(self, image_u8, ksize_range, inv, seed):
        from PIL import Image
        np.random.seed(seed)
        return np.asarray(self.dpr.Dilate(ksize_range, inv)(Image.fromarray(image_u8)))

    def custom_dataset_item(self, h5path, idx):
        return tuple(t.numpy() for t in self.dpr.CustomDataset(h5path)[idx])

    # ---- loss.py (parts that need no pretrained VGG19 download) --------------------------------
    def loss_align_obj(self, output, target, limit=32):
        return self.dpr.CombinedLoss.align_obj(torch.from_numpy(output).clone(), torch.from_numpy(target), limit).numpy()

    def loss_grad(self, output, target):
        return float(self.dpr.CombinedLoss.grad_loss(torch.from_numpy(output), torch.from_numpy(target)))

    def loss_fourier(self, output, target, log):
        return float(self.dpr.CombinedLoss.fourier_loss(torch.from_numpy(output), torch.from_numpy(target), log))

    # ---- scheduler.py / asam.py ------------------------------------------------------------------
    def scheduler_lrs(self, steps, **kwargs):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        sch = self.dpr.CosineAnnealingWarmUpRestarts(opt, **kwargs)
        lrs = []
        for _ in range(steps):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        return np.asarray(lrs)

    def minimizer_step(self, kind, w1, b1, w2, x, y, rho, eta=0.01):
        model = torch.nn.Sequential(torch.nn.Linear(w1.shape[1], w1.shape[0]), torch.nn.Tanh(),
                                    torch.nn.Linear(w2.shape[1], w2.shape[0], bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(w1)); model[0].bias.copy_(torch.from_numpy(b1))
            model[2].weight.copy_(torch.from_numpy(w2))
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mini = (self.dpr.ASAM(opt, model, rho=rho, eta=eta) if kind == "ASAM" else self.dpr.SAM(opt, model, rho=rho))
        loss_fn = lambda: torch.mean((model(torch.from_numpy(x)) - torch.from_numpy(y)) ** 2)
        loss_fn().backward()
        mini.ascent_step()
        loss_fn().backward()
        mini.descent_step()
        return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}

    # ---- weightedpartialconv2d.py ----------------------------------------------------------------
    def weighted_pconv(self, weight, bias, x, mask, weight_model, object_size=64):
        conv = self.WeightedPartialConv2d(weight.shape[1], weight.shape[0], weight.shape[-1], 1, weight.shape[-1] // 2,
                                          bias=bias is not None, return_mask=True, weight_model=weight_model,
                                          object_size=object_size)
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(weight))
            if bias is not None:
                conv.bias.copy_(torch.from_numpy(bias))
            out, m = conv(torch.from_numpy(x), torch.from_numpy(mask))
        return out.numpy(), m.numpy()
