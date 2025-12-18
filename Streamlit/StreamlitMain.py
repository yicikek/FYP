# Streamlit/StreamlitMain.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from models.shared_encoder import SharedEffB3Encoder
from models.dmad_model import DMAD_SiameseArcFace
from models.smad_model import load_smad_model
from utils.transforms import test_transform
from utils.cosine_threshold import cosine_decision
from utils.gradcam import GradCAM
from utils.gradcam_classifier import GradCAMClassifier
import cv2
import numpy as np


# -------------------------------
# 🔧 DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_MEAN = [0.485, 0.456, 0.406]
PRE_STD  = [0.229, 0.224, 0.225]

def denormalize(img_tensor):
    mean = torch.tensor(PRE_MEAN).view(3,1,1).to(img_tensor.device)
    std  = torch.tensor(PRE_STD).view(3,1,1).to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0,1)
def make_overlay(cam, img_tensor, img_pil):
    cam = cv2.resize(cam, (img_pil.width, img_pil.height))

    img_denorm = denormalize(img_tensor[0])
    img_denorm = img_denorm.permute(1,2,0).detach().cpu().numpy()
    img_denorm = cv2.resize(img_denorm, (img_pil.width, img_pil.height))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = np.clip(0.65 * heatmap + 0.35 * img_denorm, 0, 1)
    return overlay





# -------------------------------
# 🔧 MODEL LOADING (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    # Shared encoder for D-MAD (loads S-MAD weights for better features)
    # 🔹 Build encoder exactly like training
    encoder = SharedEffB3Encoder(
        weights_path="/workspaces/FYP/Streamlit/weights/efficientnet_smad_finetuned.pth",
        freeze_backbone=False
    ).to(device)

    # 🔹 Build D-MAD model with same hyperparameters as training
    dmad_model = DMAD_SiameseArcFace(
        encoder,
        embed_dim=512,
        id_fusion_weight=0.5
    ).to(device)

    # 🔹 Load D-MAD checkpoint (your trained weights)
    checkpoint = torch.load(
        "/workspaces/FYP/Streamlit/weights/dmad_checkpoint_Latest.pth",
        map_location=device,
        weights_only=False  # REQUIRED for PyTorch 2.6+
    )

    # 🔹 Some of your checkpoints are stored as {"model_state_dict": ...}
    if "model_state_dict" in checkpoint:
        dmad_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        dmad_model.load_state_dict(checkpoint)

    dmad_model.eval()
    print("✔ D-MAD model loaded successfully")


 # S-MAD EfficientNet-B3
    smad_model = load_smad_model()
    ckpt_smad = torch.load(
        "/workspaces/FYP/Streamlit/weights/efficientnet_b3_morphing.pth",
        map_location=device,
        weights_only=False,  # PyTorch 2.6 fix
    )

    try:
        smad_model.load_state_dict(ckpt_smad)
    except Exception:
        if "model_state_dict" in ckpt_smad:
            smad_model.load_state_dict(ckpt_smad["model_state_dict"])
        else:
            raise

    smad_model.to(device)
    smad_model.eval()

    for p in dmad_model.parameters():
        p.requires_grad = True

    # Grad-CAM on last conv block of EfficientNet-B3
    target_layer = smad_model.features[6]
    grad_cam_smad = GradCAM(smad_model, target_layer)
# Pick last conv layer of EfficientNet-B3 for CAM
    target_layer = dmad_model.encoder.features[8]
    grad_cam_cls = GradCAMClassifier(dmad_model, target_layer)


    return dmad_model, smad_model, grad_cam_smad, grad_cam_cls



dmad_model, smad_model, grad_cam, grad_cam_dmad = load_models()

# -------------------------------
# 🌐 UI
# -------------------------------
st.title("🔍 Morphing Attack Detection & Explainability")

st.write(
    """
Upload an **ID image** and a **selfie image**.

The system will:
- Compare them using your **D-MAD model** (classifier + cosine similarity)
- Show **bona vs morph confidence**
- Display **cosine similarity** and apply a threshold
- Generate a **heatmap** from the S-MAD model highlighting suspicious regions
"""
)

col1, col2 = st.columns(2)
with col1:
    uploaded_id = st.file_uploader("📎 Upload ID Image", type=["jpg", "jpeg", "png"])
with col2:
    uploaded_selfie = st.file_uploader("📎 Upload Selfie Image", type=["jpg", "jpeg", "png"])

# Sidebar threshold control
st.sidebar.header("⚙️ Verification Settings")
threshold = st.sidebar.slider(
    "Cosine Threshold (lower = more strict, morph if similarity < threshold)",
    min_value=-1.00,
    max_value=1.00,
    value=0.93,
    step=0.001,
)

if uploaded_id is not None and uploaded_selfie is not None:
    # -------------------------------
    # 🖼️ Load & show original images
    # -------------------------------
    img_id = Image.open(uploaded_id).convert("RGB")
    img_selfie = Image.open(uploaded_selfie).convert("RGB")

    st.subheader("👀 Input Images")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_id, caption="ID Image", use_container_width=True)
    with c2:
        st.image(img_selfie, caption="Selfie Image", use_container_width=True)

    # -------------------------------
    # 🔄 Preprocess
    # -------------------------------
    t_id = test_transform(img_id).unsqueeze(0).to(device)
    t_selfie = test_transform(img_selfie).unsqueeze(0).to(device)

    # -------------------------------
    # 🚀 Run D-MAD model
    # -------------------------------
    with torch.no_grad():
        logits, cosine_sim = dmad_model(t_id, t_selfie)

    probs = F.softmax(logits, dim=1)[0]
    bona_conf = probs[0].item()
    morph_conf = probs[1].item()
    cls_pred = int(torch.argmax(probs).item())

    cos_val, cos_pred = cosine_decision(cosine_sim, threshold)

    # -------------------------------
    # 📊 Show numeric results
    # -------------------------------
    st.subheader("📌 Model Outputs")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Classifier Evaluation**")
        st.write(f"- Bona-fide confidence: `{bona_conf:.4f}`")
        st.write(f"- Morph confidence:     `{morph_conf:.4f}`")
        st.write(f"- Predicted class:      `{'Morph' if cls_pred == 1 else 'Bona-fide'}`")

    with col_b:
        st.markdown("**Cosine Similarity (D-MAD Embeddings)**")
        st.write(f"- Cosine similarity: `{cos_val:.4f}`")
        st.write(f"- Threshold:         `{threshold:.4f}`")
        st.write(f"- Cosine decision:   `{'Morph' if cos_pred == 1 else 'Bona-fide'}`")

    # -------------------------------
    # 🧠 Fused Decision
    # -------------------------------
    st.subheader("🧩 Fused Decision")

    if cos_pred == 1:
        final_label = "❌ MORPH ATTACK DETECTED"
        st.error(final_label)
    elif morph_conf > 0.7:   # optional safety margin
        final_label = "❌ MORPH ATTACK DETECTED"
        st.error(final_label)
    else:
        final_label = "✅ BONA-FIDE USER"
        st.success(final_label)




    st.subheader("🔥 Grad-CAM Visualization (Paper Style)")
    t_id_cam = test_transform(img_id).unsqueeze(0).to(device)
    t_selfie_cam = test_transform(img_selfie).unsqueeze(0).to(device)

    t_id_cam.requires_grad_(True)
    t_selfie_cam.requires_grad_(True)

    cam_id = grad_cam.generate(t_id_cam, class_idx=1)       # Morph CAM
    cam_selfie = grad_cam.generate(t_selfie_cam, class_idx=1)

    overlay_id = make_overlay(cam_id, t_id_cam, img_id)
    overlay_selfie = make_overlay(cam_selfie, t_selfie_cam, img_selfie)

        # -------- TOP ROW: Original Images --------
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_id, caption="ID Image", use_container_width=True)
    with c2:
        st.image(img_selfie, caption="Selfie Image", use_container_width=True)

    # -------- BOTTOM ROW: Grad-CAM --------
    c3, c4 = st.columns(2)
    with c3:
        st.image(overlay_id, caption="ID Grad-CAM", use_container_width=True)
    with c4:
        st.image(overlay_selfie, caption="Selfie Grad-CAM", use_container_width=True)

