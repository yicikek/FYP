import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SharedEffB3Encoder(nn.Module):
    def __init__(self, weights_path=None, freeze_backbone=False):
        super().__init__()

        # Same architecture as S-MAD
        base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, 2)

        if weights_path is not None:
            state = torch.load(weights_path, map_location=device)
            base_model.load_state_dict(state)
            print("✅ Loaded S-MAD weights into EfficientNet-B3 encoder")

        # Use only feature extractor + pooling
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        # For EfficientNet-B3, this should be 1536
        self.out_dim = num_features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)   # (B, 1536)
        return x
