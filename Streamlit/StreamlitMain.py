# Streamlit/StreamlitMain.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

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
# 🧠 PAGE STATE INIT (REQUIRED)
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Hybrid"

st.markdown("""
<style>
    /* Dark Theme Backgrounds */
    [data-testid="stSidebar"] {
        background-color: #0b1220;
    }
    .stApp {
        background-color: #0d1117;
    }

    /* Style Sidebar Buttons to look like Nav Tabs */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 5px;
        background-color: transparent;
        color: #e5e7eb;
        border: none;
        
        /* Centering Logic */
        display: flex;
        justify-content: center; /* Centers horizontally */
        align-items: center;     /* Centers vertically */
        text-align: center;      
        
        padding: 12px 15px;
        font-weight: 500;
        transition: all 0.3s;
    }
            
    /* Active/Selected Styling (Optional: add a border or background for the active one) */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:focus, 
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:active {
        background-color: #1f2937;
        color: white;
        border: 1px solid #30363d;
    }

    /* Active/Hover Button Styling */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:hover {
        background-color: #1f2937;
        color: white;
    }
            

    /* Highlight Active Page (This is tricky in Streamlit, 
       we'll handle it by applying a style if the session_state matches) */
    
    /* Center and Style Headers */
    h1 {
        text-align: left;
        font-weight: 700;
        color: #ffffff;
    }

    /* File Uploader Container */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #30363d;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


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

BASE_DIR = os.path.dirname(__file__) 

# -------------------------------
# 🔧 MODEL LOADING (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    # Shared encoder for D-MAD (loads S-MAD weights for better features)
    # 🔹 Build encoder exactly like training
    encoder = SharedEffB3Encoder(
        weights_path = os.path.join(BASE_DIR, "weights", "efficientnet_b3_morphing.pth"),
        freeze_backbone=False
    ).to(device)

    # 🔹 Build D-MAD model with same hyperparameters as training
    dmad_model = DMAD_SiameseArcFace(
        encoder,
        embed_dim=512,
        id_fusion_weight=0.5
    ).to(device)

    # 🔹 Load D-MAD checkpoint (your trained weights)
    dmad_path = os.path.join(BASE_DIR, "weights", "Final_dmad_checkpoint.pth")
    checkpoint = torch.load(
        dmad_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        dmad_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        dmad_model.load_state_dict(checkpoint)
    dmad_model.eval()
    print("✔ D-MAD model loaded successfully")

    # 🔹 S-MAD EfficientNet-B3
    smad_model = load_smad_model()
    smad_path = os.path.join(BASE_DIR, "weights", "Final_efficientnet_smad_finetuned.pth")
    ckpt_smad = torch.load(
        smad_path,
        map_location=device,
        weights_only=False,
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

    # 🔹 S-MAD Morphing model
    smad_model1 = load_smad_model()
    smad1_path = os.path.join(BASE_DIR, "weights", "efficientnet_b3_morphing.pth")
    ckpt_smad1 = torch.load(
        smad1_path,
        map_location=device,
        weights_only=False,
    )

    try:
        smad_model1.load_state_dict(ckpt_smad1)
    except Exception:
        if "model_state_dict" in ckpt_smad1:
            smad_model1.load_state_dict(ckpt_smad1["model_state_dict"])
        else:
            raise

    smad_model.to(device)
    smad_model.eval()

    for p in dmad_model.parameters():
        p.requires_grad = True

    # Grad-CAM on last conv block of EfficientNet-B3
    target_layer = smad_model1.features[6]
    grad_cam_smad = GradCAM(smad_model1, target_layer)
# Pick last conv layer of EfficientNet-B3 for CAM
    target_layer = dmad_model.encoder.features[6]
    grad_cam_cls = GradCAMClassifier(dmad_model, target_layer)


    return dmad_model, smad_model, grad_cam_smad, grad_cam_cls



dmad_model, smad_model, grad_cam, grad_cam_dmad = load_models()

# -------------------------------
# 🧭 SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.markdown("### 🔍 Detection Modes")

modes = {
    "Hybrid Morphing Attack Detection": "Hybrid",
    "Single Morphing Attack Detection": "SMAD",
    "Differential Morphing Attack Detection": "DMAD"
}

for label, key in modes.items():
    prefix = "🌟 " if st.session_state.page == key else "  "
    if st.sidebar.button(f"{prefix}{label}", key=f"btn_{key}"):
        st.session_state.page = key
        st.rerun()

# -------------------------------
# 🖼️ PAGE CONTENT LOGIC
# -------------------------------

# --- HYBRID MODE PAGE ---
if st.session_state.page == "Hybrid":
    st.title("🧩 Hybrid Face Morphing Attack Detection")
    st.write("Combined S-MAD and D-MAD analysis for maximum security.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_id = st.file_uploader("📎 Upload ID Image", type=["jpg", "jpeg", "png"], key="hybrid_id")
    with col2:
        uploaded_selfie = st.file_uploader("📎 Upload Selfie Image", type=["jpg", "jpeg", "png"], key="hybrid_selfie")

    if uploaded_id and uploaded_selfie:
        # 🖼️ Load & show original images

        img_id = Image.open(uploaded_id).convert("RGB")
        img_selfie = Image.open(uploaded_selfie).convert("RGB")
        
        # -------------------------------
        # 🔄 Preprocess
        # -------------------------------
        t_id = test_transform(img_id).unsqueeze(0).to(device)
        t_selfie = test_transform(img_selfie).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_id = smad_model(t_id)
            probs_id = F.softmax(logits_id, dim=1)[0]

        smad_bona = probs_id[0].item()
        smad_morph = probs_id[1].item()

        # -------------------------------
        # 🚀 Run D-MAD model
        # -------------------------------
        
        with torch.no_grad():
            _, cosine_sim = dmad_model(t_id, t_selfie)
        DMAD_THRESHOLD = 0.798
        # ✅ Use the SAME function as ipynb
        cos_val, cos_pred = cosine_decision(cosine_sim, DMAD_THRESHOLD)

        


        # -------------------------------
        # 📊 Show numeric results
        # -------------------------------
        st.subheader("📌 Model Outputs")
        SMAD_THRESHOLD = 0.6
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("🆔 S-MAD Result (ID Image)")
            st.write(f"- Bona-fide probability: `{smad_bona:.4f}`")
            st.write(f"- Morph probability: `{smad_morph:.4f}`")
            st.write(f"- Cosine decision:   `{'Morph' if smad_morph > SMAD_THRESHOLD else 'Bona-fide'}`")


        with col_b:
            st.markdown("🔁 D-MAD Result (ID vs Selfie)")
            st.write(f"- Cosine similarity: `{cos_val:.4f}`")
            st.write(f"- Threshold: `{DMAD_THRESHOLD:.4f}`")
            st.write(f"- Cosine decision:   `{'Morph' if cos_pred == 1 else 'Bona-fide'}`")

        # -------------------------------
        # 🧠 Fused Decision
        # -------------------------------
        st.subheader("🧩 Fused Decision")

        # Logic flags
        smad_rejected = smad_morph > SMAD_THRESHOLD
        dmad_rejected = cos_val < DMAD_THRESHOLD

        if smad_rejected and dmad_rejected:
            st.error("### ❌ REJECTED: High Morph Risk")
            st.write("Both models have flagged this attempt as a potential Morphing Attack.")

        elif smad_rejected:
            st.warning("### ⚠️ REJECTED: S-MAD Flag")
            st.write("The ID image shows patterns consistent with digital morphing (S-MAD).")

        elif dmad_rejected:
            st.warning("### ⚠️ REJECTED: D-MAD Flag")
            st.write("The ID and Selfie do not match sufficiently, indicating a potential identity mismatch (D-MAD).")

        else:
            st.success("### ✅ ACCEPTED: Bona-fide User")
            
    


        if dmad_rejected:
            st.subheader("🔥 Analysis of Suspicious Regions (Grad-CAM Visualization)")

            # --- Grad-CAM Processing ---
            t_id_cam = t_id.clone().detach().requires_grad_(True)
            t_selfie_cam = t_selfie.clone().detach().requires_grad_(True)

            # Using the fixed generate function from our previous step
            cam_id = grad_cam_dmad.generate(t_id_cam, t_selfie_cam, class_idx=1)
            cam_selfie = grad_cam_dmad.generate(t_selfie_cam, t_id_cam, class_idx=1)

            overlay_id = make_overlay(cam_id, t_id_cam, img_id)
            overlay_selfie = make_overlay(cam_selfie, t_selfie_cam, img_selfie)

            # --- Grad-CAM Display (Smaller) ---
            # Using the same spacer logic to keep it consistent
            sp3, c3, c4, sp4 = st.columns([1, 2, 2, 1])
            with c3:
                st.image(overlay_id, caption="ID Grad-CAM", use_container_width=True)
            with c4:
                st.image(overlay_selfie, caption="Selfie Grad-CAM", use_container_width=True)

        elif smad_rejected :
            st.subheader("🔥 Analysis of Suspicious Regions (Grad-CAM Visualization)")

            # --- Grad-CAM Processing ---
            t_id_cam = t_id.clone().detach().requires_grad_(True)
            t_selfie_cam = t_selfie.clone().detach().requires_grad_(True)

            # Using the fixed generate function from our previous step
            cam_id = grad_cam.generate(t_id_cam, class_idx=1)


            overlay_id = make_overlay(cam_id, t_id_cam, img_id)


            # --- Grad-CAM Display (Smaller) ---
            # Using the same spacer logic to keep it consistent
            sp3, c3, c4, sp4 = st.columns([1, 2, 2, 1])
            with c3:
                st.image(overlay_id, caption="ID Grad-CAM", use_container_width=True)

        else:
            st.subheader("👀 Input Images")
            st.success("✅ Identity Verified. No suspicious artifacts detected.")
            # Using spacers [left_spacer, img1, img2, right_spacer]
            sp1, c1, c2, sp2 = st.columns([1, 2, 2, 1]) 
            with c1:
                st.image(img_id, caption="ID Image", use_container_width=True)
            with c2:
                st.image(img_selfie, caption="Selfie Image", use_container_width=True)

           

# --- S-MAD ONLY PAGE ---
elif st.session_state.page == "SMAD":
    st.title("🆔 S-MAD Detection")
    st.write("Analyzing ID image for morphing artifacts.")
    
    uploaded_id = st.file_uploader("📎 Upload ID Image", type=["jpg", "jpeg", "png"], key="smad_id")

    if uploaded_id:
        img_id = Image.open(uploaded_id).convert("RGB")
        t_id = test_transform(img_id).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = F.softmax(smad_model(t_id), dim=1)[0]
        
        morph_prob = probs[1].item()
        is_morph = morph_prob > 0.7

        # 2. Result Banner
        if is_morph:
            st.error(f"🚨 RESULT: MORPH DETECTED (Confidence: {morph_prob:.4f})")
        else:
            st.success(f"✅ RESULT: BONA-FIDE (Confidence: {probs[0].item():.4f})")


        # 1. Smaller Grad-CAM (using spacers)
        st.subheader("🔥 Suspicious Region Heatmap")
        t_id_cam = t_id.clone().detach().requires_grad_(True)
        cam = grad_cam.generate(t_id_cam, class_idx=1)
        overlay = make_overlay(cam, t_id_cam, img_id)
        
        col_left, col_mid, col_right = st.columns([1, 2, 1]) # Makes the middle column smaller
        with col_mid:
            st.image(overlay, caption="S-MAD Analysis", use_container_width=True)


# --- D-MAD ONLY PAGE ---
elif st.session_state.page == "DMAD":
    st.title("🔁 D-MAD Detection")
    st.write("Comparing features between ID and Selfie images.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_id = st.file_uploader("📎 Upload ID", type=["jpg", "jpeg", "png"], key="dmad_id")
    with col2:
        uploaded_selfie = st.file_uploader("📎 Upload Selfie", type=["jpg", "jpeg", "png"], key="dmad_selfie")

    # 🔹 Load D-MAD checkpoint
    checkpoint = torch.load(
        "/workspaces/FYP/Streamlit/weights/Final_dmad_checkpoint.pth",
        map_location=device,
        weights_only=False
    )

    # 🛠️ FIX: Extract the weight dictionary if it's wrapped
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Now load it into the model
    dmad_model.load_state_dict(state_dict)
    dmad_model.eval()

    # In your UI logic:
    if uploaded_id and uploaded_selfie:

        img_id = Image.open(uploaded_id).convert("RGB")
        img_selfie = Image.open(uploaded_selfie).convert("RGB")
        
        t_id = test_transform(img_id).unsqueeze(0).to(device)
        t_selfie = test_transform(img_selfie).unsqueeze(0).to(device)

        # 1. Calculate similarity
        with torch.no_grad():
            _, cos_sim = dmad_model(t_id, t_selfie)
        cos_val, pred = cosine_decision(cos_sim, 0.798)

        # 4. Final Result Banner
        if pred == 1:
            st.error(f"🚨 RESULT: MORPH DETECTED (Similarity: {cos_val:.4f})")
        else:
            st.success(f"✅ RESULT: BONA-FIDE (Similarity: {cos_val:.4f})")

        # 2. Generate Grad-CAM (Requires gradients)
        t_id_cam = t_id.clone().detach().requires_grad_(True)
        t_selfie_cam = t_selfie.clone().detach().requires_grad_(True)
        
        # Target the shared encoder to see features used for comparison
        cam_id = grad_cam_dmad.generate(t_id_cam, t_selfie_cam, class_idx=1)
        cam_selfie = grad_cam_dmad.generate(t_selfie_cam, t_id_cam, class_idx=1)

        # 3. Small & Professional Layout
        st.subheader("🔥 Comparison Heatmaps")
        spacer1, col1, col2, spacer2 = st.columns([0.5, 2, 2, 0.5]) # Makes them smaller
        with col1:
            st.image(make_overlay(cam_id, t_id_cam, img_id), caption="ID Features")
        with col2:
            st.image(make_overlay(cam_selfie, t_selfie_cam, img_selfie), caption="Selfie Features")

        