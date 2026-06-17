import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1


def load_base_model():
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=512, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=512, out_features=9, bias=True),
    )
    return model


def load_finetuned_model(device):
    model = load_base_model()
    model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
    return model
