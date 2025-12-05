import torch.nn.functional as F

def cosine_decision(cosine, threshold):
    cos_val = float(cosine.item())
    pred = 1 if cos_val < threshold else 0
    return cos_val, pred
