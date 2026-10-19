import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """
    Small CNN baseline for CIFAR-10.
    Architecture:
    - Conv 3->32, kernel 3
    - BatchNorm
    - ReLU
    - MaxPool 2x2
    - Conv 32->64, kernel 3
    - BatchNorm
    - ReLU
    - MaxPool 2x2
    - Flatten
    - Linear -> 128
    - ReLU
    - Dropout
    - Linear -> 10
    """
    def __init__(self):
        super(SmallCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # After two 2x2 pooling layers, 32x32 image becomes 8x8
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
