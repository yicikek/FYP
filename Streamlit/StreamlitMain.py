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
        margin=0.5,
        scale=30.0
    ).to(device)

    # 🔹 Load D-MAD checkpoint (your trained weights)
    checkpoint = torch.load(
        "/workspaces/FYP/Streamlit/weights/dmad_checkpoint_0.6 (1).pth",
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
    min_value=0.70,
    max_value=0.99,
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
        st.markdown("**Classifier (ArcFace Head)**")
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

    if (morph_conf > bona_conf) and (cos_pred == 1):
        final_label = "❌ MORPH ATTACK DETECTED"
        st.error(final_label)
    else:
        final_label = "✅ BONA-FIDE USER"
        st.success(final_label)




   ############### 🔥 RUN GRAD-CAM ON S-MAD ###############
    st.subheader("🔥 Grad-CAM Heatmap (Hybrid Morphing Attack Detection)")

    # Make sure NOT to use no_grad() for S-MAD
    t_id_cam = test_transform(img_id).unsqueeze(0).to(device)
    t_id_cam.requires_grad_(True)

    # Run S-MAD forward pass WITH gradients
    smad_logits = smad_model(t_id_cam)

    # 1 = morph class (change if reversed)
    cam_map = grad_cam.generate(t_id_cam, class_idx=1)

    # Resize CAM to selfie size
    cam_resized = cv2.resize(cam_map, (img_id.width, img_id.height))

    # Convert to heatmap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay (0.6*image + 0.4*heatmap)
    img_np = np.array(img_id).astype(np.float32)
    overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)

    st.image(overlay, caption="S-MAD Grad-CAM Heatmap", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img_id, caption="Original Selfie")
    with c2:
        st.image(overlay, caption="S-MAD Grad-CAM Heatmap", use_container_width=True)