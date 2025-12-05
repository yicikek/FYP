# models/smad_model.py
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def load_smad_model(num_classes: int = 2):
    """
    Returns an EfficientNet-B3 model configured for S-MAD:
    - 2-class classifier (bona vs morph)
    """
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model
