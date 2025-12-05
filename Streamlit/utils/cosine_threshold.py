# utils/cosine_threshold.py
def cosine_decision(cosine_tensor, threshold: float):
    """
    cosine_tensor: tensor of shape (B,) or (1,)
    threshold: decision boundary
    Returns: (cos_value, pred_label)
    - pred_label = 1 → morph
    - pred_label = 0 → bona
    """
    value = float(cosine_tensor.squeeze().item())
    pred = 1 if value > threshold else 0
    return value, pred
