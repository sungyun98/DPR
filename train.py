"""Train the DPR network with DistributedDataParallel on one or more nodes.

The training and validation sets are the HDF5 files written by ``generate_dataset.ipynb``
(``./datasets/dataset_train_n96k.h5`` and ``./datasets/dataset_valid_n12k.h5``). Rank 0
writes ``./checkpoint.pt`` every 10 epochs, from which a restarted run resumes, and
``./model_min.pt`` whenever the validation loss reaches a new minimum after epoch 120.

Launch with torchrun, one process per GPU::

    torchrun
        --nnodes=$NUM_NODES
        --nproc_per_node=$NUM_GPU
        --node_rank=${0 to $NUM_NODES-1 for each node}
        --max-restarts=$NUM_ALLOWED_FAILURES
        --rdzv_id=$JOB_ID
        --rdzv_endpoint=$HOST_NODE_ADDR:$PORT
        train.py $EPOCHS_TOTAL

Example command for each node (NODE01, NODE02, and NODE03 in order)::

    [@NODE01]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=0 --rdzv_id=123 \
        --rdzv_endpoint=NODE01:29400 train.py 600 &
    [@NODE02]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=1 --rdzv_id=123 \
        --rdzv_endpoint=NODE01:29400 train.py 600 > /dev/null &
    [@NODE03]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=2 --rdzv_id=123 \
        --rdzv_endpoint=NODE01:29400 train.py 600 > /dev/null &
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from deeppr import ASAM, CombinedLoss, CosineAnnealingWarmUpRestarts, CustomDataset, Network


def ddp_setup() -> None:
    """Initialize the NCCL process group and select the GPU of this process (LOCAL_RANK)."""
    init_process_group(backend="nccl")
    torch.backends.cudnn.benchmark = True  # enable cuDNN library benchmark
    # for debugging, enable anomaly detection with torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    """Distributed training loop with ASAM, gradient clipping and checkpoints.

    The loss is `CombinedLoss` on the output with ``false_scale=True``. The learning rate
    follows `CosineAnnealingWarmUpRestarts` (restart period ``epochs_total // 15``, doubling).

    Parameters
    ----------
    model : torch.nn.Module
        Network to train; moved to the GPU of this process and wrapped in DDP.
    dl_train, dl_valid : torch.utils.data.DataLoader
        Training and validation loaders with a `DistributedSampler`.
    optimizer : torch.optim.Optimizer
        Optimizer of the model parameters.
    epochs_total : int
        Total number of epochs.
    step_ckp : int
        Interval in epochs between checkpoints.
    path_ckp : str
        Checkpoint path; training resumes from it if it exists.
    compile_model : bool, default False
        Compile the DDP-wrapped model with `torch.compile`. On one RTX 6000 Ada GPU the
        training step was 1.2 times faster with a third less memory (after about 4 minutes
        of compilation); on 4 GPUs with SyncBatchNorm, whose communication splits the
        compiled graph, epochs were slower (7.1 s instead of 6.2 s), hence off by default.
    """

    def __init__(
        self,
        model: nn.Module,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        optimizer: Optimizer,
        epochs_total: int,
        step_ckp: int,
        path_ckp: str,
        compile_model: bool = False,
    ) -> None:

        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        if self.global_rank == 0:
            print(f"[{time.ctime()}] Starting training with {self.world_size} GPUs")

        self.model = model.to(self.local_rank)
        self.dl_train = dl_train
        self.dl_valid = dl_valid

        self.criterion = CombinedLoss(coeffs=(1, 10, 0.1, 0.01), device=self.local_rank)

        self.optimizer = optimizer
        self.grad_clip = 1  # max_norm for gradient clipping (alternative: 0.1)

        # sharpness-aware minimizer: ASAM, or SAM(self.optimizer, model, rho=0.1) from deeppr,
        # or None for plain optimizer steps
        self.minimizer = ASAM(self.optimizer, model, rho=0.2, eta=1e-2)

        self.epochs_total = epochs_total
        self.epochs_run = 0
        self.loss_hist = None
        self.step_ckp = step_ckp
        self.path_ckp = path_ckp
        if os.path.exists(self.path_ckp):
            self._load_checkpoint(self.path_ckp)

        # LR scheduler: cosine annealing with warm restarts, or MultiStepLR(self.optimizer,
        # milestones=[500], gamma=0.1, last_epoch=self.epochs_run - 1) from torch.optim, or None
        scheduler_kwargs = {
            "T_0": self.epochs_total // 15,
            "T_mult": 2,
            "T_up": 1,
            "eta_min": 1e-8,
            "eta_max_0": 5e-3,
            "gamma": 1,
        }
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer, **scheduler_kwargs, last_epoch=self.epochs_run - 1
        )

        self.model = DDP(model, device_ids=[self.local_rank])
        if compile_model:
            # attributes such as no_sync() and module reach the DDP model through the wrapper
            self.model = torch.compile(self.model)

    def _run_batch(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """Compute the loss of one batch and, if ``train``, update the model."""
        self.optimizer.zero_grad()
        input = input * mask
        output = self.model(input, mask, false_scale=True)

        if train:
            loss = self.criterion(output, target)

            if self.minimizer is not None:
                with self.model.no_sync():
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip
                    )  # gradient clipping
                self.minimizer.ascent_step()
                self.criterion(self.model(input, mask, false_scale=True), target).backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )  # gradient clipping
                self.minimizer.descent_step()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )  # gradient clipping
                self.optimizer.step()

        else:
            loss = self.criterion(output, target)

        return loss.data

    def _run_epoch(self, epoch: int) -> torch.Tensor:
        """Train and validate for one epoch; return the mean losses ``[train, valid]``."""
        # train
        self.model.train()
        self.dl_train.sampler.set_epoch(epoch)
        loss_train = torch.zeros(1, device=self.local_rank)
        for input, target, mask in self.dl_train:
            input = input.to(self.local_rank)
            target = target.to(self.local_rank)
            mask = mask.to(self.local_rank)
            loss_train += self._run_batch(input, target, mask, train=True)
        loss_train /= len(self.dl_train)

        # validation
        self.model.eval()
        self.dl_valid.sampler.set_epoch(epoch)
        loss_valid = torch.zeros(1, device=self.local_rank)
        with torch.no_grad():
            for input, target, mask in self.dl_valid:
                input = input.to(self.local_rank)
                target = target.to(self.local_rank)
                mask = mask.to(self.local_rank)
                loss_valid += self._run_batch(input, target, mask, train=False)
        loss_valid /= len(self.dl_valid)

        return torch.cat((loss_train, loss_valid))

    def _save_checkpoint(self, epoch: int) -> None:
        """Save the epoch, loss history, model and optimizer states to ``path_ckp``."""
        torch.save(
            {
                "epochs_run": epoch,
                "loss_hist": self.loss_hist,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.path_ckp,
        )
        print(f"[{time.ctime()}] Saving checkpoint at {self.path_ckp}")

    def _save_model_only(self, epoch: int, path_model: str = "./model.pt") -> None:
        """Save the epoch and the model state (the format of ``pretrained/``)."""
        torch.save(
            {"epochs_run": epoch, "model_state_dict": self.model.module.state_dict()}, path_model
        )

    def _load_checkpoint(self, path_ckp: str) -> None:
        """Restore the epoch, loss history, model and optimizer states."""
        loc = f"cuda:{self.local_rank}"
        ckp = torch.load(path_ckp, map_location=loc, weights_only=True)

        self.epochs_run = ckp["epochs_run"] + 1
        self.loss_hist = ckp["loss_hist"]
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])

        if self.global_rank == 0:
            print(f"[{time.ctime()}] Resuming training from checkpoint at Epoch {self.epochs_run}")

    def train(self) -> None:
        """Run the remaining epochs, averaging the losses over all processes."""
        loss_valid_min = 100
        for epoch in range(self.epochs_run, self.epochs_total):
            t0 = time.time()

            loss = self._run_epoch(epoch)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.world_size

            if self.loss_hist is None:
                self.loss_hist = torch.zeros(2, self.epochs_total, device=self.local_rank)
            if self.loss_hist.shape[-1] < self.epochs_total:
                epochs_prev = self.loss_hist.shape[-1]
                self.loss_hist = torch.cat(
                    (
                        self.loss_hist,
                        torch.zeros(
                            2, self.epochs_total - self.loss_hist.shape[-1], device=self.local_rank
                        ),
                    ),
                    dim=-1,
                )
                if self.global_rank == 0:
                    print(
                        f"[{time.ctime()}] Total epochs changed from {epochs_prev} "
                        f"to {self.epochs_total}"
                    )

            self.loss_hist[:, epoch] = loss

            lr = self.optimizer.param_groups[0]["lr"]

            if self.global_rank == 0:
                print(
                    f"[{time.ctime()}] Epoch {epoch + 1}/{self.epochs_total} | "
                    f"Train loss: {loss[0]:.6f} | Valid loss: {loss[1]:.6f} | "
                    f"Learning rate: {lr:.6f} | Elapsed time: {time.time() - t0:.1f} s"
                )

                if (epoch + 1) % self.step_ckp == 0:
                    self._save_checkpoint(epoch)

                if (epoch + 1) >= 120 and loss[1] < loss_valid_min:
                    loss_valid_min = loss[1]
                    self._save_model_only(epoch, path_model="./model_min.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        if self.global_rank == 0:
            print(f"[{time.ctime()}] Training for total {self.epochs_total} epochs finished")


def prepare_train(batch_size: int) -> tuple[DataLoader, DataLoader, nn.Module, Optimizer]:
    """Create the data loaders, the network (with synchronized BatchNorm) and AdamW.

    Parameters
    ----------
    batch_size : int
        Batch size per process.

    Returns
    -------
    dl_train, dl_valid : torch.utils.data.DataLoader
        Training and validation loaders.
    model : torch.nn.Module
        DPR network.
    optimizer : torch.optim.Optimizer
        AdamW optimizer.
    """
    dset_train = CustomDataset(h5path="./datasets/dataset_train_n96k.h5")
    dset_valid = CustomDataset(h5path="./datasets/dataset_valid_n12k.h5")
    dl_train = DataLoader(
        dataset=dset_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dset_train),
    )

    dl_valid = DataLoader(
        dataset=dset_valid,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dset_valid),
    )

    model = Network(
        ngf=64, max_features=1024, weight_model=True, downsample_FFC=False, refinement=True
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # DDP sync for BatchNorm layer

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4
    )

    return dl_train, dl_valid, model, optimizer


def main(
    epochs_total: int,
    batch_size: int,
    step_ckp: int,
    path_ckp: str = "./checkpoint.pt",
    compile_model: bool = False,
) -> None:
    """Set up DDP, train and clean up.

    Parameters
    ----------
    epochs_total : int
        Total number of epochs.
    batch_size : int
        Batch size per process.
    step_ckp : int
        Interval in epochs between checkpoints.
    path_ckp : str, default './checkpoint.pt'
        Checkpoint path.
    compile_model : bool, default False
        Compile the model with `torch.compile` (see `Trainer`).
    """
    ddp_setup()
    dl_train, dl_valid, model, optimizer = prepare_train(batch_size)
    trainer = Trainer(
        model, dl_train, dl_valid, optimizer, epochs_total, step_ckp, path_ckp, compile_model
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    epochs_total = int(sys.argv[1])

    main(epochs_total, batch_size=16, step_ckp=10)
