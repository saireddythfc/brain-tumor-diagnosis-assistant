import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # -> [B, 32, 299, 299]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 32, 149, 149]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 149, 149]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 64, 74, 74]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> [B, 128, 74, 74]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 128, 37, 37]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> [B, 256, 37, 37]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> [B, 256]
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
