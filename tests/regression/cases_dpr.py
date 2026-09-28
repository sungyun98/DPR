"""Regression cases for DPR (canonical NumPy interface, see adapters/).

Synthetic inputs come from ``numpy.random.RandomState(seed)``, whose streams are stable
across NumPy versions, so later PyTorch versions are tested on identical data. The only
case that depends on the PyTorch random number generator is ``dataset_generate_diffraction``
(``GenerateDiffraction`` draws its noise with torch); see ``RNG_DEPENDENT``.
"""

import os
import tempfile

import numpy as np
from harness import capture_error
from scipy.io import loadmat
from scipy.ndimage import binary_dilation, gaussian_filter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_PATTERN = os.path.join(REPO, "exp", "20170321_3480_35_ag_flower.mat")
N_SEED = 2
N_ITER = 40
CROP = (slice(224, 288), slice(224, 288))  # the 64 x 64 object region of a 512 x 512 frame
CKPTS = ["param_dpr0.pt", "param_dpr1.pt"]
PARAM_R = {
    "algorithm": "GPS-R",
    "error": "R",
    "shrinkwrap": False,
    "sigma": (0, 0.1, 0.4, 1),
    "alpha_count": 5,
    "t": 1,
    "s": 0.9,
}  # demo.ipynb refinement settings


# ---- synthetic data -----------------------------------------------------------------------
def synthetic_object(seed, size=64, extent=40):
    """Smooth random blob with ~extent px diameter, non-negative, in a size x size frame."""
    rs = np.random.RandomState(seed)
    field = gaussian_filter(rs.randn(size, size), 4)
    yy, xx = np.mgrid[:size, :size] - size / 2 + 0.5
    envelope = np.exp(-(((yy**2 + xx**2) / (extent / 2.4) ** 2) ** 2))
    shape = (field * envelope > 0.02 * np.abs(field).max()) * envelope
    density = 0.5 + gaussian_filter(rs.rand(size, size), 2)
    return (shape * density).astype(np.float32)


def synthetic_pattern(seed, photons=3e6):
    """Noisy centred diffraction pattern of a synthetic object and its validity mask."""
    obj = synthetic_object(seed)
    frame = np.zeros((512, 512), np.float64)
    frame[CROP] = obj
    inten = np.abs(np.fft.fftshift(np.fft.fft2(frame))) ** 2
    inten *= photons / inten.sum()
    rs = np.random.RandomState(seed + 1000)
    inten = rs.poisson(inten) + rs.normal(0, 1 / 2.35482, inten.shape)
    mask = np.ones((512, 512), bool)
    mask[256 - 8 : 256 + 8, 256 - 8 : 256 + 8] = False  # beam stop
    mask[:, 300:306] = False  # detector gap
    return inten.astype(np.float32)[None, None], mask[None, None], obj


def pr_inputs(seed=31):
    inten, mask, obj = synthetic_pattern(seed)
    inten, mask = inten[0, 0], mask[0, 0]
    amplitude = np.sqrt(np.fft.ifftshift(np.clip(inten, 0, None) * mask)).astype(np.float32)
    unknown = np.fft.ifftshift(~mask).astype(np.float32)
    frame = np.zeros((512, 512), bool)
    frame[CROP] = obj > 0
    support = binary_dilation(frame, iterations=2).astype(np.float32)
    return amplitude, support, unknown


def random_phase(seed, shape):
    theta = np.random.RandomState(seed).rand(*shape) * 2 * np.pi
    return np.exp(1j * theta).astype(np.complex64)


# ---- centred FFT helpers ----------------------------------------------------------------------
def case_fft_helpers(api):
    rs = np.random.RandomState(1)
    z = (rs.randn(2, 1, 16, 12) + 1j * rs.randn(2, 1, 16, 12)).astype(np.complex64)
    x = rs.randn(2, 1, 15, 13).astype(np.float32)
    return {
        "fft2c_complex": api.fft2c(z),
        "ifft2c_complex": api.ifft2c(z),
        "fft2c_real_odd": api.fft2c(x),
        "ifft2c_real_odd": api.ifft2c(x),
    }


# ---- phase retrieval (module/phaseretrieval.py) --------------------------------------------------
SHRINKWRAP = dict(
    shrinkwrap=True, sigma_initial=3, sigma_limit=1.5, ratio_update=0.01, threshold=0.1, interval=10
)
GPS_COMMON = dict(sigma=(0, 0.01, 0.4, 0.1, 0.7, 1), alpha_count=10, t=1, s=0.8)
PR_CONFIGS = {
    "HIO": dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0.2),
    "RAAR_step": dict(
        algorithm="RAAR", error="R", beta=0.75, beta_type="step", beta_lim=1, boundary_push=0
    ),
    "RAAR_linear_NLL": dict(
        algorithm="RAAR", error="NLL", beta=0.5, beta_type="linear", beta_lim=1, boundary_push=0.1
    ),
    "RAAR_schedule": dict(
        algorithm="RAAR", error="R", beta=(0, 0.5, 0.5, 0.9), beta_type="const", boundary_push=0
    ),
    "GPS-R": dict(algorithm="GPS-R", error="R", **GPS_COMMON),
    "GPS-F": dict(algorithm="GPS-F", error="R", **GPS_COMMON),
    "HIO_shrinkwrap": dict(
        algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0.2, **SHRINKWRAP
    ),
    "GPS-R_shrinkwrap": dict(algorithm="GPS-R", error="R", **dict(GPS_COMMON, **SHRINKWRAP)),
}


def make_pr_case(name):
    def case(api):
        amplitude, support, unknown = pr_inputs()
        info = dict(error="R", shrinkwrap=False)
        info.update(PR_CONFIGS[name])
        phase = random_phase(10, (N_SEED, 512, 512))
        res = capture_error(api.phase_retrieval, amplitude, support, unknown, info, N_ITER, phase)
        if isinstance(res, str):
            return {"result": res}
        out, path = res
        outside = np.array(out)
        outside[(slice(None),) + CROP] = 0
        return {
            "u_crop": out[(slice(None),) + CROP],
            "u_outside_abs_sum": float(np.abs(outside).sum()),
            "path": path,
        }

    case.__name__ = "case_pr_" + name
    return case


# ---- pretrained network -----------------------------------------------------------------------------
def case_network_experimental(api):
    inp, mask = api.demo()["prepare_data"](np.array(loadmat(EXP_PATTERN)["pattern"]), bin=None)
    inp, mask = inp.numpy(), mask.numpy()
    out = {"input_sum": float(inp.sum()), "mask_sum": int(mask.sum())}
    for ckpt in CKPTS:
        out["out_" + ckpt[:-3]] = api.network(ckpt, inp, mask)
    out["out_param_dpr1_false_scale"] = api.network("param_dpr1.pt", inp, mask, false_scale=True)
    return out


def case_network_synthetic(api):
    out = {}
    for seed in [21, 22]:
        inp, mask, _ = synthetic_pattern(seed)
        for ckpt in CKPTS:
            out[f"out_{ckpt[:-3]}_seed{seed}"] = api.network(ckpt, inp, mask)
    return out


def case_demo_pipeline(api):
    res = api.demo_pipeline(loadmat(EXP_PATTERN)["pattern"], "param_dpr1.pt", PARAM_R, 50)
    return {
        "input_sum": float(res["input"].sum()),
        "mask_sum": int(res["mask"].sum()),
        "center_shift": res["center_shift"].astype(np.int64),
        "dpr": res["dpr"],
        "dpr_refined": res["dpr_refined"],
        "refine_error": res["refine_error"],
    }


# ---- dataset.py ----------------------------------------------------------------------------------
def case_dataset_transforms(api):
    img = (
        (gaussian_filter(np.random.RandomState(2).rand(128, 128), 3) * 255 / 0.62)
        .clip(0, 255)
        .astype(np.uint8)
    )
    binary = api.binarize(img, 0.6)
    return {
        "binarize": binary.astype(np.uint8),
        "dilate_inv_9_49": api.dilate(binary, (9, 49), True, seed=0).astype(np.uint8),
        "dilate_3_7": api.dilate(binary, (3, 7), False, seed=1).astype(np.uint8),
        "dilate_0_0": api.dilate(binary, (0, 0), False, seed=2).astype(np.uint8),
    }


def case_dataset_generate_diffraction(api):
    obj = np.stack([synthetic_object(s) for s in (41, 42)])[:, None]
    inten, target = api.generate_diffraction(obj, seed=0, ph_ord=6, l_coh=200, false_scale=True)
    return {
        "intensity_sample0": inten[0, 0],
        "target": target,
        "total_photons": inten.sum(axis=(1, 2, 3)),
    }


def case_dataset_custom_h5(api):
    import h5py

    rs = np.random.RandomState(3)
    data = {
        "input": rs.rand(3, 1, 8, 8).astype(np.float32),
        "target": rs.rand(3, 1, 4, 4).astype(np.float32),
        "mask": rs.rand(3, 1, 8, 8) > 0.5,
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.h5")
        with h5py.File(path, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)
        inp, tgt, msk = api.custom_dataset_item(path, 1)
    return {"input": inp, "target": tgt, "mask": msk}


# ---- loss.py (no pretrained VGG19 needed) ---------------------------------------------------------
def case_loss_components(api):
    target = np.stack([synthetic_object(s) for s in (51, 52)])[:, None]
    output = np.roll(target, (3, -2), axis=(-2, -1)).copy()
    output[1] = np.rot90(output[1], 2, axes=(-2, -1))
    output = output + 0.01 * np.random.RandomState(4).randn(*output.shape).astype(np.float32)
    output = output.astype(np.float32)
    return {
        "align_obj": api.loss_align_obj(output, target, 32),
        "grad_loss": api.loss_grad(output, target),
        "fourier_loss_log": api.loss_fourier(output, target, True),
        "fourier_loss_amp": api.loss_fourier(output, target, False),
    }


# ---- scheduler.py / asam.py ---------------------------------------------------------------------------
def case_scheduler(api):
    return {
        "train_py_config": api.scheduler_lrs(
            200, T_0=40, T_mult=2, T_up=1, eta_min=1e-8, eta_max_0=5e-3, gamma=1
        ),
        "mult1_gamma0.5": api.scheduler_lrs(
            50, T_0=10, T_mult=1, T_up=3, eta_min=0, eta_max_0=0.1, gamma=0.5
        ),
    }


def case_minimizers(api):
    rs = np.random.RandomState(7)
    w1, b1, w2 = (
        (rs.randn(8, 4) * 0.5).astype(np.float32),
        (rs.randn(8) * 0.1).astype(np.float32),
        (rs.randn(2, 8) * 0.5).astype(np.float32),
    )
    x, y = rs.randn(16, 4).astype(np.float32), rs.randn(16, 2).astype(np.float32)
    out = {}
    for kind, rho in [("ASAM", 0.2), ("SAM", 0.1)]:
        for k, v in api.minimizer_step(kind, w1, b1, w2, x, y, rho).items():
            out[f"{kind}_{k}"] = v
    return out


# ---- weightedpartialconv2d.py ------------------------------------------------------------------------------
def case_weighted_pconv(api):
    rs = np.random.RandomState(8)
    weight, bias = rs.randn(4, 1, 3, 3).astype(np.float32), rs.randn(4).astype(np.float32)
    x, mask = (
        rs.rand(1, 1, 32, 32).astype(np.float32),
        (rs.rand(1, 1, 32, 32) > 0.3).astype(np.float32),
    )
    out = {}
    for tag, wm in [("guinier_porod", True), ("uniform", False)]:
        o, m = api.weighted_pconv(weight, bias, x, mask, weight_model=wm, object_size=8)
        out["out_" + tag], out["mask_" + tag] = o, m.astype(bool)
    return out


# ---- registry ------------------------------------------------------------------------------------------------
CASES = {"fft_helpers": case_fft_helpers}
for _name in PR_CONFIGS:
    CASES["pr_" + _name] = make_pr_case(_name)
for _fn in [
    case_network_experimental,
    case_network_synthetic,
    case_demo_pipeline,
    case_dataset_transforms,
    case_dataset_generate_diffraction,
    case_dataset_custom_h5,
    case_loss_components,
    case_scheduler,
    case_minimizers,
    case_weighted_pconv,
]:
    CASES[_fn.__name__[5:]] = _fn

TOLERANCE = {name: {"*": 1e-6} for name in CASES}
for _name in CASES:
    if _name.startswith("pr_") or _name == "demo_pipeline":
        TOLERANCE[_name] = {"*": 1e-4}
TOLERANCE["network_experimental"] = TOLERANCE["network_synthetic"] = {"*": 1e-5}
# demo.ipynb works in float32 only, so its float32 noise was measured by perturbing the experimental
# pattern by one float32 epsilon (relative, 3 seeds, legacy code and environment): dpr changed by
# <= 2.8e-6, dpr_refined (50 GPS-R iterations) by <= 8.5e-4, refine_error by <= 8.6e-6. Tolerances
# are three times these values.
TOLERANCE["demo_pipeline"] = {"*": 1e-4, "dpr": 1e-5, "dpr_refined": 2.5e-3, "refine_error": 3e-5}

# Cases with float64 references (references_f64/, see generate_f64_references.py). The demo
# pipeline and networks are excluded: demo.ipynb creates float32 tensors explicitly.
F64_CASES = ["pr_HIO", "pr_RAAR_step", "pr_RAAR_schedule", "pr_GPS-R", "pr_GPS-F"]
F64_DROP = []
F64_TOLERANCE = 1e-9
F32_NOISE_FACTOR = 3  # float32 tolerance >= this factor x the legacy code's own float32 error

RNG_DEPENDENT = {
    "dataset_generate_diffraction": "GenerateDiffraction draws coherence, flux and Poisson/Gaussian noise from the "
    "torch RNG; exact values are only reproducible with the same torch version",
}
_SW_CENTRED = (
    "the shared phaseretrieval package centres the ShrinkWrap Gaussian kernel; DPR's copy used an "
    "ifftshifted kernel that split the Gaussian into lobes about +-ceil(2 * sigma_initial) px apart"
)
EXPECTED_CHANGES = {
    "pr_RAAR_linear_NLL": "DPR's copy summed the NLL over five dimensions of a 4-D tensor and raised IndexError; "
    "the shared phaseretrieval package sums over (1, 2, 3)",
    "pr_HIO_shrinkwrap": _SW_CENTRED,
    "pr_GPS-R_shrinkwrap": _SW_CENTRED,
}
