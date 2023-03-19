from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 10

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adamax(model.parameters(), lr=0.003, weight_decay=0.0007)

    transforms = Compose([ToTensor()])
