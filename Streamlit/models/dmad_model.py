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
    """
    Improved Proposed Model
    - Keeps morph detection logic unchanged
    - Improves bonafide compactness
    - Fully compatible with existing train_dmad
    """
    def __init__(self, shared_encoder, embed_dim=512, id_fusion_weight=0.5):
        super().__init__()
        self.encoder = shared_encoder
        self.id_fusion_weight = id_fusion_weight

        out_dim = self.encoder.out_dim

        # 🔹 Artifact head (PRIMARY — drives morph detection)
        self.fc_art = nn.Linear(out_dim, embed_dim)

        # 🔹 Identity head (AUXILIARY — stabilises bonafide)
        self.fc_id = nn.Linear(out_dim, embed_dim)

        # 🔹 Classifier (unchanged interface)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # bona / morph
        )

    def forward(self, img1, img2, labels=None):
        # ============================
        # Encode images
        # ============================
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)

        # ============================
        # Artifact embeddings (USED for morph)
        # ============================
        e1_art = F.normalize(self.fc_art(f1), dim=1)
        e2_art = F.normalize(self.fc_art(f2), dim=1)

        # ============================
        # Identity embeddings (USED for bonafide stability)
        # ============================
        e1_id = F.normalize(self.fc_id(f1), dim=1)
        e2_id = F.normalize(self.fc_id(f2), dim=1)

        # ============================
        # Pair embeddings
        # ============================
        pair_art = (e1_art - e2_art) ** 2
        pair_id  = (e1_id  - e2_id ) ** 2

        # 🔑 Identity fused ONLY into representation
        pair_embed = pair_art + self.id_fusion_weight * pair_id

        # ============================
        # Outputs (UNCHANGED API)
        # ============================
        logits = self.classifier(pair_embed)

        # 🔑 Cosine similarity used for:
        # - contrastive loss
        # - evaluation
        # (artifact head ONLY → morph behaviour unchanged)
        cosine_sim = F.cosine_similarity(e1_art, e2_art, dim=1)

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
