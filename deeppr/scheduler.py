"""Learning-rate scheduler: cosine annealing with linear warm-up and warm restarts."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmUpRestarts(LRScheduler):
    """Cosine annealing with warm restarts, a linear warm-up and decaying peaks.

    Cycle ``i`` lasts ``T_i = T_0 * T_mult**i`` epochs: the learning rate rises linearly from
    ``eta_min`` to ``eta_max_0 * gamma**i`` in ``T_up`` epochs, then decreases to ``eta_min``
    along a cosine. The learning rates of the optimizer are overwritten (its initial ``lr``
    is not used).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    T_0 : int
        Length of the first cycle in epochs (positive).
    T_mult : int, default 1
        Factor of the cycle length after each restart (at least 1).
    T_up : int, default 0
        Warm-up epochs at the start of each cycle (smaller than ``T_0``).
    eta_min : float, default 0
        Minimum learning rate.
    eta_max_0 : float, default 0.1
        Peak learning rate of the first cycle.
    gamma : float, default 1
        Factor of the peak learning rate per cycle.
    last_epoch : int, default -1
        Index of the last epoch, to resume training.

    Raises
    ------
    ValueError
        If ``T_0``, ``T_mult`` or ``T_up`` is not an integer in the allowed range.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        T_up: int = 0,
        eta_min: float = 0,
        eta_max_0: float = 0.1,
        gamma: float = 1,
        last_epoch: int = -1,
    ) -> None:
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected positive integer T_up, but got {T_up}")

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_up = T_up

        self.eta_min = eta_min
        self.eta_max_0 = eta_max_0
        self.eta_max_i = eta_max_0
        self.gamma = gamma

        self.T_cur = last_epoch
        self.cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Return the learning rate of each parameter group for the current epoch."""
        if self.T_cur == -1:
            return [self.eta_min for _ in self.base_lrs]
        elif self.T_cur < self.T_up:
            return [
                self.eta_min + (self.eta_max_i - self.eta_min) * self.T_cur / self.T_up
                for _ in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (self.eta_max_i - self.eta_min)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)))
                / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch: float | None = None) -> None:
        """Advance to the next epoch, or to ``epoch``, and set the learning rates.

        Parameters
        ----------
        epoch : float, optional
            Epoch to go to (non-negative); by default the next one.

        Raises
        ------
        ValueError
            If ``epoch`` is negative.
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch < 0:
            raise ValueError(f"Expected non-negative epoch, but got {epoch}")

        if epoch >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = epoch % self.T_0
                self.T_i = self.T_0
                self.cycle = epoch // self.T_0
            else:
                n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult**n
                self.cycle = n
        else:
            self.T_cur = epoch
            self.T_i = self.T_0
            self.cycle = 0
        self.eta_max_i = self.eta_max_0 * (self.gamma**self.cycle)

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
