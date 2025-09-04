import torch.nn as nn
from torchvision import models

def build_resnet50_head(num_classes, pretrained=True, dropout_p=0.5):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(512, num_classes)
    )
    return model