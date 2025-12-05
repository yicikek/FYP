import os
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    select_largest=True,
    post_process=False,
    min_face_size=32,
    thresholds=[0.4, 0.5, 0.5],
    keep_all=False,
    device='cuda:0'
)

def brighten_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def align_images(in_folder, out_folder, failed_folder):
    print('Processing images in:', in_folder)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)
    skipped_imgs = []

    for img_name in tqdm(os.listdir(in_folder)):
        filepath = os.path.join(in_folder, img_name)
        img = cv2.imread(filepath)
        if img is None:
            skipped_imgs.append(img_name)
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

        # Try to brighten dark images
        if boxes is None or landmarks is None:
            bright_img = brighten_image(img)
            rgb_bright = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = mtcnn.detect(rgb_bright, landmarks=True)
            if boxes is None or landmarks is None:
                cv2.imwrite(os.path.join(failed_folder, img_name), img)
                skipped_imgs.append(img_name)
                continue
            img = bright_img  # use enhanced version

        x1, y1, x2, y2 = boxes[0]
        n = 0.08  # margin
        w, h = x2 - x1, y2 - y1
        x1 -= n * w
        y1 -= n * h
        x2 += n * w
        y2 += n * h

        # Ensure valid cropping
        h_img, w_img = img.shape[:2]
        x1 = int(max(0, min(w_img - 1, x1)))
        y1 = int(max(0, min(h_img - 1, y1)))
        x2 = int(max(0, min(w_img, x2)))
        y2 = int(max(0, min(h_img, y2)))

        # Skip if invalid or zero-size
        if x2 <= x1 or y2 <= y1:
            cv2.imwrite(os.path.join(failed_folder, img_name), img)
            skipped_imgs.append(img_name)
            continue

        crop = img[y1:y2, x1:x2]

        # Validate final crop
        if crop is None or crop.size == 0:
            cv2.imwrite(os.path.join(failed_folder, img_name), img)
            skipped_imgs.append(img_name)
            continue

        cv2.imwrite(os.path.join(out_folder, img_name), crop)

    print(f"Images with no Face: {len(skipped_imgs)}")
    if skipped_imgs:
        print("Skipped:", skipped_imgs)

def main():
    base_in = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL"
    base_out = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_preprocessed"
    failed_out = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_failed"

    folders = {"morphed": "morphed"}
    for in_sub, out_sub in folders.items():
        align_images(
            os.path.join(base_in, in_sub),
            os.path.join(base_out, out_sub),
            os.path.join(failed_out, out_sub)
        )

if __name__ == "__main__":
    main()
