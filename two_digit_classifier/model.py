import os
import certifi
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class TwoDigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        os.environ["SSL_CERT_FILE"] = certifi.where()
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove original fully connected layer
        self.backbone.fc = nn.Identity()

        # New classification heads
        self.digit1_head = nn.Linear(512, 10)  # First digit: 0-9
        self.digit2_head = nn.Linear(512, 11)  # Second digit: 0-9 + "empty" (class 10)

    def forward(self, x):
        features = self.backbone(x)
        return self.digit1_head(features), self.digit2_head(features)

