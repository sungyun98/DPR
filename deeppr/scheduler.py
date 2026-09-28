import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, T_up=0, eta_min=0, eta_max_0=0.1, gamma=1, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        
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
        
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return [self.eta_min
                    for _ in self.base_lrs]
        elif self.T_cur < self.T_up:
            return [self.eta_min + (self.eta_max_i - self.eta_min) * self.T_cur / self.T_up
                    for _ in self.base_lrs]
        else:
            return [self.eta_min + (self.eta_max_i - self.eta_min) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for _ in self.base_lrs]

    def step(self, epoch=None):
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
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** n
                self.cycle = n
        else:
            self.T_cur = epoch
            self.T_i = self.T_0
            self.cycle = 0
        self.eta_max_i = self.eta_max_0 * (self.gamma ** self.cycle)
        
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr