import torch

PRE_MEAN = [0.485, 0.456, 0.406]
PRE_STD  = [0.229, 0.224, 0.225]

def denormalize(img_tensor):
    mean = torch.tensor(PRE_MEAN).view(3,1,1)
    std  = torch.tensor(PRE_STD).view(3,1,1)
    return (img_tensor * std + mean).clamp(0,1)
