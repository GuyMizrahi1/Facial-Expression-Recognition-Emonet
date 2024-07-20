import math
import torch
from torch.optim.lr_scheduler import LRScheduler


# The file implements the learning rate scheduler, which defines how the learning rate changes over time during training
class CosineAnnealingWithWarmRestartsLR(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, warmup_steps: int = 128,
                 cycle_steps: int = 512, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.max_lr = initial_lr

        self.steps_counter = 0

        super().__init__(optimizer)

    def step(self, epoch=None):
        self.steps_counter += 1

        current_cycle_steps = self.steps_counter % self.cycle_steps

        if current_cycle_steps < self.warmup_steps:
            current_lr = (self.min_lr + (self.max_lr - self.min_lr) * current_cycle_steps / self.warmup_steps)
        else:
            current_lr = (self.min_lr + (self.max_lr - self.min_lr) *
                          (1 + math.cos(math.pi * (current_cycle_steps - self.warmup_steps) /
                                        (self.cycle_steps - self.warmup_steps))
                           ) / 2
                          )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
