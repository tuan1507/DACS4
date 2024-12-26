import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorCNN(nn.Module):
    def __init__(self, num_classes):
        super(ColorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, 64 * 16 * 16)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
