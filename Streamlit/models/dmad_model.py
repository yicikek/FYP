# models/dmad_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features=2, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # input: (B, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (B, out_features)

        if label is None:
            # Inference mode (no margin)
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(label, num_classes=self.out_features).float().to(input.device)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class DMAD_SiameseArcFace(nn.Module):
    def __init__(self, shared_encoder, embed_dim=512, margin=0.5, scale=30.0):
        super().__init__()
        self.encoder = shared_encoder



        # Project 1536 → 512-d embedding
        self.fc_embed = nn.Linear(self.encoder.out_dim, embed_dim)

        # ArcFace head on pair embedding
        self.arcface = ArcMarginProduct(in_features=embed_dim,
                                        out_features=2,
                                        m=margin,
                                        s=scale)

    def forward(self, img1, img2, labels=None):
        # Shared encoder
        f1 = self.encoder(img1)   # (B, 1536)
        f2 = self.encoder(img2)   # (B, 1536)

        e1 = F.normalize(self.fc_embed(f1))  # (B, 512)
        e2 = F.normalize(self.fc_embed(f2))  # (B, 512)

        # Cosine similarity between embeddings
        cosine_sim = F.cosine_similarity(e1, e2)  # (B,)

        # Pair embedding for ArcFace (absolute difference)
        pair_embed = torch.abs(e1 - e2)  # (B, 512)

        if labels is not None:
            logits = self.arcface(pair_embed, labels.long())
        else:
            logits = self.arcface(pair_embed, None)

        return logits, cosine_sim

def contrastive_loss_cosine(cosine_sim, labels, margin=0.5):
    """
    labels: 0 (same/bonafide), 1 (different/morph)
    cosine_sim: higher means more similar
    """
    # Convert labels to float
    labels = labels.float()

    # For genuine pairs (label=0) → want cosine_sim → 1
    pos_loss = (1 - labels) * (1.0 - cosine_sim) ** 2

    # For morph pairs (label=1) → want cosine_sim < margin
    neg_loss = labels * torch.clamp(cosine_sim - margin, min=0) ** 2

    loss = (pos_loss + neg_loss).mean()
    return loss
