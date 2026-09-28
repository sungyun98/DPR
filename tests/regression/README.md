# Numerical regression suite

Checks that changes to the code do not alter numerical results beyond floating-point
tolerance. The reference outputs in `references/` were generated from the original code
(tag `v1.0-legacy`) with the library versions of the original README (PyTorch 2.1, CPU
builds); see `environment-legacy.yml`.

## Layout

| File | Purpose |
| --- | --- |
| `cases_dpr.py` | Test cases with tolerances |
| `adapters/` | One adapter per implementation, translating between NumPy arrays and that implementation's API |
| `harness.py` | Saving, loading and comparing results |
| `generate_references.py` | Writes `references/<case>.npz` and `references/<case>.json` |
| `test_regression.py` | pytest entry point comparing an implementation with the references |

## Covered

- Phase retrieval algorithms bundled with DPR (HIO, RAAR with beta schedules, GPS-R/F,
  shrinkwrap, R and NLL errors) on a synthetic 512 x 512 pattern
- Pretrained networks `param_dpr0.pt` and `param_dpr1.pt` on the experimental pattern
  `exp/20170321_3480_35_ag_flower.mat` (prepared as in `demo.ipynb`) and on two synthetic
  patterns
- The complete `demo.ipynb` pipeline (DPR, GPS-R refinement, alignment), using the helper
  functions from the notebook's own code cell
- Centred FFT helpers, weighted partial convolution, dataset transforms, `GenerateDiffraction`,
  `CustomDataset`, the loss terms that need no pretrained VGG19 download (object alignment,
  gradient loss, Fourier loss), the learning-rate scheduler, and one ASAM/SAM step

The VGG19 perceptual loss is not covered: it downloads pretrained ImageNet weights at run
time.

Inputs come from `numpy.random.RandomState(seed)` (stable across NumPy versions) or from the
experimental pattern. Only `dataset_generate_diffraction` depends on the PyTorch random number
generator (`RNG_DEPENDENT` in `cases_dpr.py`).

## Running

```bash
git worktree add --detach ../_legacy/DPR v1.0-legacy
REG_IMPL=dpr_legacy REG_CODE_ROOT=../_legacy/DPR \
    conda run -n dpr-legacy python -m pytest tests/regression -q
```

A result passes when its relative L2 difference `||new - ref|| / ||ref||` is within the case
tolerance in `cases_dpr.py` (1e-6 for direct operations, 1e-5 for the networks, 1e-4 for
iterative algorithms and the demo pipeline). NaN positions and error types must match exactly.

## Known behaviour of the original code recorded in the references

- `pr_RAAR_linear_NLL`: the NLL error metric raises `IndexError` (it sums over five dimensions
  of a four-dimensional tensor).

Intended changes of results are listed with their reason in `EXPECTED_CHANGES`
(`cases_dpr.py`); such cases are reported as expected failures.
