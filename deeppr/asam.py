# Adaptive Sharpness-Aware Minimization (ASAM) from https://github.com/SamsungLabs/ASAM
#   (asam.py, archived at Software Heritage swh:1:rev:f156a680171db16d551c0d85cba2514fa3bff6a2)
#   Copyright 2021 Samsung Research, Apache License 2.0 (LICENSES/ASAM-Apache-2.0.txt)
#   Reformatted with ruff (including the import order); docstrings and type hints added;
#   otherwise unmodified.

"""Sharpness-aware minimizers: SAM and adaptive SAM (ASAM)."""

from collections import defaultdict

import torch
from torch import nn
from torch.optim import Optimizer


class ASAM:
    """Adaptive sharpness-aware minimization (ASAM).

    Each training step is a two-step update: `ascent_step` moves the weights to the
    (scale-adaptive) worst case within a neighbourhood of radius ``rho``, and
    `descent_step` restores them and applies the optimizer with the gradient computed there.
    Call ``loss.backward()`` before each of the two steps.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Base optimizer of the model parameters.
    model : torch.nn.Module
        Model being trained.
    rho : float, default 0.5
        Radius of the neighbourhood.
    eta : float, default 0.01
        Offset added to the weight magnitudes of the adaptive scaling.

    References
    ----------
    .. [1] J. Kwon et al., ASAM: adaptive sharpness-aware minimization for scale-invariant
       learning of deep neural networks, ICML 2021, https://arxiv.org/abs/2102.11600
    """

    def __init__(
        self, optimizer: Optimizer, model: nn.Module, rho: float = 0.5, eta: float = 0.01
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self) -> None:
        """Perturb the weights towards the worst case and reset the gradients."""
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if "weight" in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if "weight" in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self) -> None:
        """Restore the weights, apply the optimizer step and reset the gradients."""
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    """Sharpness-aware minimization (SAM): `ASAM` without the adaptive scaling.

    Parameters
    ----------
    optimizer, model, rho
        As for `ASAM`; ``eta`` is not used.

    References
    ----------
    .. [1] P. Foret et al., Sharpness-aware minimization for efficiently improving
       generalization, ICLR 2021, https://arxiv.org/abs/2010.01412
    """

    @torch.no_grad()
    def ascent_step(self) -> None:
        """Perturb the weights towards the worst case and reset the gradients."""
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
