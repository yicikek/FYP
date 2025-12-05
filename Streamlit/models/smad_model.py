import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def load_smad_model():
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model
