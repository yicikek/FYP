import os
from PIL import Image

import os
from PIL import Image
import numpy as np

def compute_average_ratio(folder):
    ratios = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png")):
            img = Image.open(os.path.join(folder, f))
            w, h = img.size
            ratios.append(w / h)
    return np.mean(ratios)

avg_ratio = compute_average_ratio("FEI_preprocessed/bonafide")
print("Average FEI ratio =", avg_ratio)


input_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_preprocessed"
output_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_resized"

os.makedirs(output_dir, exist_ok=True)

TARGET_SIZE = (124, 224)

for folder in ["bonafide", "morphed"]:
    in_path = os.path.join(input_dir, folder)
    out_path = os.path.join(output_dir, folder)
    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(in_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = Image.open(os.path.join(in_path, img_name)).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.BICUBIC)
        img.save(os.path.join(out_path, img_name), "JPEG", quality=90)

print("✔ FEI images resized and saved at:", output_dir)


FEI_RATIO = 1.33     # average 4:3 ratio
TARGET_H  = 256
TARGET_W  = int(TARGET_H * FEI_RATIO)  # ≈ 341

train_transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])