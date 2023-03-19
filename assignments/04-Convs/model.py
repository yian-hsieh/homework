import torch


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1000, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
