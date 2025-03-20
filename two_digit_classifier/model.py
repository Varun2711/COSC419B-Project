import os
import certifi
import torch.nn as nn
from torchvision import models


class TwoDigitClassifier(nn.Module):
    def __init__(self, model_arch="resnet18"):
        super().__init__()

        os.environ["SSL_CERT_FILE"] = certifi.where()

        model_fn = getattr(models, model_arch)
        self.backbone = model_fn(weights="DEFAULT")

        # Remove original fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # New classification heads
        # Fisrt digit: 0-9
        # Second digit: 0-9 + "empty" (class 10)
        self.digit1_head = nn.Linear(in_features, 10)
        self.digit2_head = nn.Linear(in_features, 11)

    def forward(self, x):
        features = self.backbone(x)
        return self.digit1_head(features), self.digit2_head(features)


def create_model(model_arch, device):
    return TwoDigitClassifier(model_arch).to(device)

