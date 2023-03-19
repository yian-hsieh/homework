import torch


class Model(torch.nn.Module):
    """
    A convolutional neural network with 5 convolutional layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 5 * 5, 1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the neurnal net.
        """
        return self.network(x)
