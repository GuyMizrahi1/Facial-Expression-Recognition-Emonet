import math
import torch
from torch.optim.lr_scheduler import LRScheduler


# The file implements the learning rate scheduler, which defines how the learning rate changes over time during training
class CosineAnnealingWithWarmRestartsLR(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, t_0: int, t_mult: int = 1.1, eta_min: float = 0.0,
                 last_epoch: int = -1, initial_lr: float = 0.0001):
        self.T_0 = t_0
        self.T_mult = t_mult
        self.eta_min = eta_min
        self.base_lr = initial_lr
        self.current_epoch = last_epoch

        super(CosineAnnealingWithWarmRestartsLR, self).__init__(optimizer, last_epoch)

    # Calculates the learning rate for the current epoch based on the cosine annealing schedule.
    def get_lr(self):
        if self.current_epoch == -1:
            return [self.base_lr for _ in self.optimizer.param_groups]
        # check if it's time to restart - it calculates a new learning rate that is closer to the base_lr
        elif (self.current_epoch - 1 - self.T_0) % (self.T_0 * self.T_mult) == 0:
            return [group['lr'] + (self.base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_0)) / 2 for group in
                    self.optimizer.param_groups]
        # calculates the learning rate based on the cosine function, oscillating between eta_min and base_lr
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.T_0)) / 2
                for base_lr in self.base_lrs]

    # Updates the learning rate for each parameter group in the optimizer.
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        # Initial Phase - the learning rate is calculated using a cosine function that depends on the current epoch.
        # This ensures a smooth transition from base_lr to eta_min
        if epoch <= self.T_0:
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_0)) / 2
        else:
            # Checks if it's time for a restart
            if (epoch - 1 - self.T_0) % (self.T_0 * self.T_mult) == 0:
                lr = self.base_lr
            else:
                # Calculates the LR using the cosine function, adjusted for the current position in the cycle.
                lr = self.eta_min + (self.base_lr - self.eta_min) * (
                        1 + math.cos(math.pi * (epoch - self.T_0) / self.T_0)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
