import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import MultiStepLR

from module import Network, CustomDataset, SAM, ASAM, CosineAnnealingWarmUpRestarts, CombinedLoss


def ddp_setup():
    init_process_group(backend='nccl')
    torch.backends.cudnn.benchmark = True # enable cuDNN library benchmark
    # torch.autograd.set_detect_anomaly(True) # enable anomaly detection
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class Trainer():
    def __init__(self, model, dl_train, dl_valid, optimizer, epochs_total, step_ckp, path_ckp):
        
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])
        if self.global_rank == 0:
            print(f'[{time.ctime()}] Starting training with {self.world_size} GPUs')
        
        self.model = model.to(self.local_rank)
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        
        self.criterion = CombinedLoss(coeffs=(1, 10, 0.1, 0.01), device=self.local_rank)
        
        self.optimizer = optimizer
        self.grad_clip = 1 # max_norm for gradient clipping
        # self.grad_clip = 0.1 # max_norm for gradient clipping
        
        # SAM
        self.minimizer = None
        # self.minimizer = SAM(self.optimizer, model, rho=0.1)
        self.minimizer = ASAM(self.optimizer, model, rho=0.2, eta=1e-2)
        
        self.epochs_total = epochs_total
        self.epochs_run = 0
        self.loss_hist = None
        self.step_ckp = step_ckp
        self.path_ckp = path_ckp
        if os.path.exists(self.path_ckp):
            self._load_checkpoint(self.path_ckp)
             
        # LR Scheduler
        self.scheduler = None
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[500], gamma=0.1,
        #                              last_epoch=self.epochs_run - 1)
        scheduler_kwargs = {'T_0': self.epochs_total // 15, 'T_mult': 2, 'T_up': 1,
                            'eta_min': 1e-8, 'eta_max_0': 5e-3, 'gamma': 1}
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, **scheduler_kwargs,
                                                       last_epoch=self.epochs_run - 1)
        
        
        self.model = DDP(model, device_ids=[self.local_rank])
        
    def _run_batch(self, input, target, mask, train=True):
        self.optimizer.zero_grad()
        input = input * mask
        output = self.model(input, mask, false_scale=True)
        
        if train:
            loss = self.criterion(output, target, align_limit=32)
            
            if self.minimizer is not None:
                with self.model.no_sync():
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip) # gradient clipping
                self.minimizer.ascent_step()
                self.criterion(self.model(input, mask, false_scale=True), target, align_limit=32).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip) # gradient clipping
                self.minimizer.descent_step()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip) # gradient clipping
                self.optimizer.step()
                
        else:
            loss = self.criterion(output, target, align_limit=32)
            
        return loss.data

    def _run_epoch(self, epoch):
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
        
    def _save_checkpoint(self, epoch):
        torch.save({'epochs_run': epoch,
                    'loss_hist': self.loss_hist,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   self.path_ckp)
        print(f'[{time.ctime()}] Saving checkpoint at {self.path_ckp}')
        
    def _save_model_only(self, epoch, path_model='./model.pt'):
        torch.save({'epochs_run': epoch,
                    'model_state_dict': self.model.module.state_dict()},
                   path_model)
        
    def _load_checkpoint(self, path_ckp):
        loc = f'cuda:{self.local_rank}'
        ckp = torch.load(path_ckp, map_location=loc)
            
        self.epochs_run = ckp['epochs_run'] + 1
        self.loss_hist = ckp['loss_hist']
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
            
        if self.global_rank == 0:
            print(f'[{time.ctime()}] Resuming training from checkpoint at Epoch {self.epochs_run}')
        
    def train(self):
        loss_valid_min = 100
        for epoch in range(self.epochs_run, self.epochs_total):
            t0 = time.time()
            
            loss = self._run_epoch(epoch)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.world_size
            
            if self.loss_hist is None:
                self.loss_hist = torch.zeros(2, self.epochs_total, device=self.local_rank)
            if self.loss_hist.shape[-1] < self.epochs_total:
                self.loss_hist = torch.cat((self.loss_hist,
                                            torch.zeros(2, self.epochs_total - self.loss_hist.shape[-1], device=self.local_rank)), dim=-1)
                if self.global_rank == 0:
                    print(f'[{time.ctime()}] Total epochs changed from {self.loss_hist.shape[-1]} to {self.epochs_total}')
                
            self.loss_hist[:, epoch] = loss
            
            lr = self.optimizer.param_groups[0]['lr']
                
            if self.global_rank == 0:
                print(f'[{time.ctime()}] Epoch {epoch + 1}/{self.epochs_total} | Train loss: {loss[0]:.6f} | Valid loss: {loss[1]:.6f} | Learning rate: {lr:.6f} | Elapsed time: {time.time() - t0:.1f} s')
                
                if (epoch + 1) % self.step_ckp == 0:
                    self._save_checkpoint(epoch)
                    
                if (epoch + 1) >= 120 and loss[1] < loss_valid_min:
                    loss_valid_min = loss[1]
                    self._save_model_only(epoch, path_model='./model_min.pt')
                
            
            if self.scheduler is not None:
                self.scheduler.step()
                
        if self.global_rank == 0:
            print(f'[{time.ctime()}] Training for total {self.epochs_total} epochs finished')
        


def prepare_train(batch_size):
    dset_train = CustomDataset(h5path='./datasets/dataset_train_n96k.h5')
    dset_valid = CustomDataset(h5path='./datasets/dataset_valid_n12k.h5')
    dl_train = DataLoader(dataset=dset_train,
                          batch_size=batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=False,
                          sampler=DistributedSampler(dset_train))
    
    dl_valid = DataLoader(dataset=dset_valid,
                         batch_size=batch_size,
                         num_workers=4,
                         pin_memory=True,
                         shuffle=False,
                         sampler=DistributedSampler(dset_valid))
    
    model = Network(ngf=64, max_features=1024, weight_model=True, downsample_FFC=False, refinement=True)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # DDP sync for BatchNorm layer
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    
    return dl_train, dl_valid, model, optimizer


def main(epochs_total, batch_size, step_ckp, path_ckp='./checkpoint.pt'):
    ddp_setup()
    dl_train, dl_valid, model, optimizer = prepare_train(batch_size)
    trainer = Trainer(model, dl_train, dl_valid, optimizer, epochs_total, step_ckp, path_ckp)
    trainer.train()
    destroy_process_group()


if __name__ == '__main__':
    import sys
    epochs_total = int(sys.argv[1])
    
    main(epochs_total, batch_size=16, step_ckp=10)
    
    '''
    torchrun
        --nnodes=$NUM_NODES
        --nproc_per_node=$NUM_GPU
        --node_rank=${0 to $NUM_GPU-1 for each node}
        --max-restarts=$NUM_ALLOWED_FAILURES
        --rdzv_id=$JOB_ID
        --rdzv_endpoint=$HOST_NODE_ADDR:$PORT
        train.py $EPOCHS_TOTAL
    
    * Example command for each node (NODE01, NODE02, and NODE03 in order)
    [@NODE01]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=0 --rdzv_id=123 --rdzv_endpoint=NODE01:29400 train.py 600 &
    [@NODE02]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=1 --rdzv_id=123 --rdzv_endpoint=NODE01:29400 train.py 600 > /dev/null &
    [@NODE03]$ nohup torchrun --nnodes=3 --nproc_per_node=4 --node_rank=2 --rdzv_id=123 --rdzv_endpoint=NODE01:29400 train.py 600 > /dev/null &
    '''
