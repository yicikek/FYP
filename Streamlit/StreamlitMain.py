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
    /* 🎨 Dynamic Backgrounds using Streamlit Variables */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    .stApp {
        background-color: var(--background-color);
    }

    /* 🧭 Sidebar Buttons (Navigation) */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 8px;
        background-color: transparent;
        color: var(--text-color); /* Dynamically changes black/white */
        border: 1px solid rgba(128, 128, 128, 0.2);
        
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 10px 10px;
        margin-bottom: 5px;
        transition: all 0.3s ease;
    }

    /* Hover & Active States */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:hover {
        background-color: var(--secondary-background-color);
        border-color: #ff4b4b; /* Streamlit Red accent */
        color: #ff4b4b;
    }

    /* 📄 File Uploader Styling */
    [data-testid="stFileUploader"] {
        background-color: var(--secondary-background-color);
        border: 1px dashed rgba(128, 128, 128, 0.3);
        border-radius: 10px;
    }

    /* 🖋️ Ensure Headers are always visible */
    h1, h2, h3 {
        color: var(--text-color) !important;
    }
    
    [data-testid="stHorizontalBlock"] [data-testid="column"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
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
    dmad_path = os.path.join(BASE_DIR, "weights", "Final_dmad_checkpoint (1).pth")
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
    dmad_model1 = DMAD_SiameseArcFace(
        encoder,
        embed_dim=512,
        id_fusion_weight=0.5
    ).to(device)
    dmad_path1 = os.path.join(BASE_DIR, "weights", "Final_dmad_checkpoint (1).pth")
    checkpoint = torch.load(
        dmad_path1,
        map_location=device,
        weights_only=False
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
    grad_cam_cls = GradCAMClassifier(dmad_model1, target_layer)


    return dmad_model, smad_model, grad_cam_smad, grad_cam_cls



dmad_model, smad_model, grad_cam, grad_cam_dmad = load_models()

with st.sidebar.expander("ℹ️ Project Information", expanded=False):
    st.markdown("""
    **Face Morphing Attack Detection**
    
    This system utilizes a **Hybrid Detection** approach to identify manipulated identity documents.
    
    * **S-MAD:** Analyzes single-image texture and digital artifacts.
    * **D-MAD:** Compares ID images against live selfies for identity consistency.
    * **Grad-CAM:** Provides explainable AI heatmaps of suspicious facial regions.
    """)

st.sidebar.divider()

# 2. Page Navigation Buttons
st.sidebar.markdown("### 📖 Main Menu")

modes = {
    "Project Overview": "Intro",  # Added this line
    "Hybrid Morphing Attack Detection": "Hybrid",
    "Single Morphing Attack Detection": "SMAD",
    "Differential Morphing Attack Detection": "DMAD"
}

# Change the default page state at the top of your script if you want it to start here:
if "page" not in st.session_state:
    st.session_state.page = "Intro"

for label, key in modes.items():
    # Highlight logic
    is_active = st.session_state.page == key
    prefix = "🌟 " if is_active else "  "
    
    if st.sidebar.button(f"{prefix}{label}", key=f"btn_{key}"):
        st.session_state.page = key
        st.rerun()

st.sidebar.divider()

# 3. System Status (Bottom of Sidebar)
st.sidebar.info(f"**Device:** {device.type.upper()}")

# -------------------------------
# 🖼️ PAGE CONTENT LOGIC


# --- INTRODUCTION PAGE ---
if st.session_state.page == "Intro":
    st.title("🛡️ Detecting Morphing Attack For Secure Digital Transaction")
        # The Subtitle (Your requested text)
    st.markdown("<h5 style='color: #000000; font-weight: 400; margin-top: -15px;'>A Hybrid Approach for Biometric Security</h5>", unsafe_allow_html=True)
    # Use columns for a professional "Key Features" look
    st.markdown("""
    <div style="
        background-color:#f1f5f9;
        padding:14px 18px;
        border-radius:8px;
        font-size:15px;
    ">
    👩‍💻 <b>Prepared by:</b> Kek Yi Ci<br>
    🎓 <b>Supervised by:</b> Dr Hoo Wai Lam
    </div>
    """, unsafe_allow_html=True)


    tab1, tab2, tab3 = st.tabs(["📖 Overview", "📂 Dataset", "🧠 Modeling"])
    with tab1:

            # ===== INTRODUCTION (FIRST 3 SLIDES) =====
        st.markdown("### ❓ What is a Face Morphing Attack?")
        st.markdown(
            "A face morphing attack blends two or more individuals’ faces into a single composite image that can resemble multiple people."
        )

        
        # ===== TWO COLUMN: PROBLEM STATEMENT & OBJECTIVE =====
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🚨 Problem Statement")
            st.markdown(
                "- Traditional face recognition relies on surface similarity and misses subtle morphing artifacts.\n"
                "- Existing S-MAD works on single images but cannot compare identity consistency.\n"
                "- Existing D-MAD compares two images but struggles with pose, lighting, and expression variations."
            )

        with col2:
            st.markdown("### 🎯 Project Objective")
            st.markdown(
                "To build a clear and reliable hybrid detection system that combines texture-based morph detection "
                "(S-MAD) and identity comparison (D-MAD) to effectively prevent face morphing attacks in eKYC systems."
            )
        
        st.markdown("### 📘 Key Concepts")

        KCtab1, KCtab2, KCtab3, KCtab4 = st.tabs(["✅ Bonafide", "🎭 Morph", "🔍 S-MAD", "🔁 D-MAD"])

        with KCtab1:
            st.markdown("""
            #### Bonafide Image
            - A **genuine, unaltered facial image**.
            - Represents **one real person**.
            - Should always pass identity verification in eKYC systems.
            """)

        with KCtab2:
            st.markdown("""
            #### Morph Image
            - Created by **blending two or more different faces** into one image.
            - May look realistic but **does not represent a real person**.
            - Can falsely match **multiple identities**, making it dangerous for eKYC.
            """)

        with KCtab3:
            st.markdown("""
            #### S-MAD (Single Morphing Attack Detection)
            - Analyzes **only one image**, usually the ID photo.
            - Detects **texture-level artifacts** caused by face blending.
            - Effective even when **no selfie or reference image** is available.
            """)

        with KCtab4:
            st.markdown("""
            #### D-MAD (Differential Morphing Attack Detection)
            - Compares **two images**: ID image and live selfie.
            - Checks **identity consistency** between the two faces.
            - Detects morphs that still look similar to real people.
            """)

        
        st.button("🚀 Start Hybrid Detection", on_click=lambda: st.session_state.update({"page": "Hybrid"}))

    with tab2:
        # ===============================
        # 📁 DATASET TAB
        # ===============================

        def resize_img(path, size=(400, 200)):
            img = Image.open(path)
            return img.resize(size)
        
        st.markdown("### 📁 Dataset Introduction")
        st.markdown(
            "Three facial datasets are used to train and evaluate the hybrid system, covering synthetic, controlled, and realistic face images for both S-MAD and D-MAD tasks."
        )
       

        dataset_tab1, dataset_tab2, eda_tab = st.tabs([
            "🔍 S-MAD Dataset (SMDD)",
            "🔁 D-MAD Dataset (FEI & FRLL)",
            "📊 EDA"
        ])
        st.divider()
        # =========================================================
        # 🔍 S-MAD TAB — SMDD DATASET
        # =========================================================
        with dataset_tab1:
            st.markdown("### 📌 SMDD Dataset (Single-Image Morph Detection)")
           
            st.markdown("""
            **SMDD (Single Morph Detection Dataset)** is used for training and evaluating  
            the **S-MAD (Single-image Morphing Attack Detection)** model.
            """)
            col_smdd,col_non = st.columns(2)

            with col_smdd:
                img_path = os.path.join(BASE_DIR, "assets", "SMDD.png")
                
                img_smdd = Image.open(img_path)
                st.image(img_smdd, caption="SMDD Dataset Example", use_container_width=True)    

            st.markdown("#### 🧩 Dataset Characteristics")
            st.markdown("""
            - Provides **separate training and testing splits**
            - Contains **bona-fide** and **synthetically generated morph images**
            - Designed specifically for **single-image morph detection**
            """)

            st.markdown("#### 👤 Image Sources")
            st.markdown("""
            - Facial images collected from **publicly available biometric datasets**
            - Images are **unaltered** and represent genuine identities
        
            """)

            st.info(
                "📌 The SMDD dataset enables the S-MAD model to learn intrinsic morphing artifacts "
                "without relying on a reference image."
            )

        # =========================================================
        # 🔁 D-MAD TAB — FEI + FRLL DATASETS
        # =========================================================
        with dataset_tab2:
            st.subheader("📌 D-MAD Datasets (Differential Morph Detection)")

            st.markdown("""
            The **D-MAD (Differential Morphing Attack Detection)** model is trained and evaluated  
            using **paired datasets**, allowing direct comparison between two facial images.
            """)

            # ---------------- FEI DATASET ----------------
                        
            st.markdown("### 1. FEI Dataset")
            
            col_FEI,col_non = st.columns(2)

            with col_FEI:
                img_path = os.path.join(BASE_DIR, "assets", "FEI_differentAngle.png")
                
                img_fei = Image.open(img_path)
                st.image(img_fei, caption="FEI Face Dataset Example", use_container_width=True)    


            fei_col1, fei_col2 = st.columns(2)

            # ---- FEI Face Dataset ----
            with fei_col1:
                st.markdown("#### 🧑 FEI Face Dataset")
                st.markdown("""
                - Contains **200 subjects** with balanced gender distribution  
                - Includes multiple **head pose variations**:
                    - frontal  
                    - rotated angles  
                - Provides **neutral and smiling expressions**
                """)

            # ---- FEI Morph Dataset ----
            with fei_col2:
                st.markdown("#### 🔀 FEI Morph Dataset")
                st.markdown("""
                **Morphing Algorithms Used:**
                - FaceFusion  
                - UTW (Universal Triangle Warping)  
                - NTNU Morphing Tool  

                **Source Dataset:** FEI Face Dataset
                """)

            st.divider()

            # =================================================
            # FRLL DATASET (FACE vs MORPH)
            # =================================================
            st.markdown("### 2. FRLL Dataset")
            col_FRLL,col_non = st.columns(2)

            with col_FRLL:
                img_path = os.path.join(BASE_DIR, "assets", "FRLL.png")
                img_frll = Image.open(img_path)

                st.image(img_frll, caption="FRLL Dataset Example", use_container_width=True) 

            frll_col1, frll_col2 = st.columns(2)

            # ---- FRLL Face Dataset ----
            with frll_col1:
                st.markdown("#### 🧑 FRLL Dataset (Face Research Lab London)")
                st.markdown("""
                - High-resolution facial images under **controlled lighting**  
                - Multiple head poses (frontal and slight rotations)  
                - Neutral and smiling expressions  
                - Clean background with **highly detailed facial features**
                """)

            # ---- ASML / FRLL Morph Dataset ----
            with frll_col2:
                st.markdown("#### 🔀 ASML / FRLL Morph Dataset")
                st.markdown("""
                **Morphing Algorithms Used:**
                - Landmark-based warping  
                - Delaunay triangulation  
                - Poisson blending  

                **Source Dataset:** FRLL (Face Research Lab London)
                """)
            
        with eda_tab:
            st.header("📊 Exploratory Data Analysis (EDA)")
            st.markdown(
                "This section analyzes the **image resolution characteristics** of each dataset "
                "to understand variability and preprocessing requirements before training."
            )
            # =================================================
            # SMDD EDA
            # =================================================
            st.subheader("📌 SMDD Dataset")

            img_smdd = Image.open(os.path.join(BASE_DIR, "assets", "EDA_SMDD.png"))
            st.image(img_smdd, use_container_width=True)

            st.markdown("""
            **Observations:**
            - Images have **uniform resolution** (approximately 256 × 256 pixels)
            - Dataset is **highly standardized**
            - No major resizing issues are expected
            """)
            st.divider()
            # =================================================
            # FEI EDA
            # =================================================
            st.subheader("📌 FEI Dataset")

            img_fei = Image.open(os.path.join(BASE_DIR, "assets", "EDA_FEI.png"))
            st.image(img_fei, use_container_width=True)

            st.markdown("""
            **Observations:**
            - Large variation in image resolution
            - Multiple clusters of widths and heights
            - **Resizing and normalization are required** before training
            """)

            st.divider()
            # =================================================
            # FRLL EDA
            # =================================================
            st.subheader("📌 FRLL Dataset")

            img_frll = Image.open(os.path.join(BASE_DIR, "assets", "EDA_FRLL.png"))
            st.image(img_frll, use_container_width=True)

            st.markdown("""
            **Observations:**
            - Images are generally **high-resolution** (around 1300 × 1300 pixels)
            - Minimal variation in width and height
            - Indicates a **clean and controlled dataset**
            """)

    with tab3 :
        st.markdown("### 🧠 Modelling Overview")

        col_mod, col_Non = st.columns(2)
        with col_mod:
            img_path = os.path.join(BASE_DIR, "assets", "Modelling.png")
            img_frll = Image.open(img_path)

            st.image(img_frll, caption="Modelling Flowchart", use_container_width=True) 

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📌 Stage 1: Train S-MAD")
            st.markdown("""
            ✅ **Objective:** Detect visual morphing artifacts from a single image  

            - Backbone: **EfficientNet-B3**
            - Learns **low-level texture & blending artifacts**
            - Trained with a **single-image classifier head**
            - Dataset: **SMDD**
            - Output: Probability of morph vs bona-fide
            """)

        with col2:
            st.markdown("### 📌 Stage 2: Train D-MAD")
            st.markdown("""
            ✅ **Objective:** Detect identity inconsistencies between two face images  

            - Reuses **encoder from Stage 1**
            - Siamese network with **shared weights**
            - Uses **angular embedding + cosine similarity**
            - Dataset: **FEI Morph + FRLL**
            - Output: Similarity score for morph decision
                        
            """)
        

        st.markdown("#### 🧠 Model Used")

        tab1, tab2, tab3 = st.tabs([
            "⚙️ EfficientNet-B3",
            "🔁 Dual-Head Siamese Network",
            "📐 Angular Embedding + Cosine Similarity"
        ])

        # -------------------------------
        # TAB 1: EfficientNet-B3
        # -------------------------------
        with tab1:

            st.markdown(
                "**EfficientNet-B3** is used as the backbone feature extractor in this system."
            )

            col1, col2 = st.columns(2)

            # ---------------- LEFT COLUMN ----------------
            with col1:
                st.markdown("#### Why EfficientNet-B3 is used:")
                st.markdown("""
                - Balances **accuracy** and **computational efficiency**
                - Learns strong **facial texture and structural features**
                - Pretrained on **ImageNet**, allowing faster convergence
                """)

            # ---------------- RIGHT COLUMN ----------------
            with col2:
                st.markdown("#### How it is used in this system:")
                st.markdown("""
                - Acts as a **shared encoder** for both **S-MAD** and **D-MAD**
                - Extracts facial **embeddings** from input images
                - Provides **consistent feature representations** across datasets
                """)

            st.markdown("#### ✅ Key Benefit")
            st.markdown(
                "High-quality feature extraction with **low computational cost**."
            )

        # -------------------------------
        # TAB 2: Dual-Head Siamese Network
        # -------------------------------
        with tab2:
            st.markdown(
                "A **Siamese Network** processes two facial images using the **same shared encoder** "
                "and compares their embeddings."
            )

            col1, col2 = st.columns(2)

            # ---------------- LEFT COLUMN ----------------
            with col1:
                st.markdown("#### Why Dual-Head Siamese Network is used:")
                st.markdown("""
                - A single embedding struggles to learn both **identity** and **morph cues**
                - Dual-head design **separates learning objectives**
                - Improves stability and task-specific learning
                """)

                st.markdown("#### Two Heads in the Network:")
                st.markdown("""
                **Artifact Head**
                - Focuses on detecting **morphing artifacts**
                - Captures blending inconsistencies
                
                **Identity Head**
                - Maintains **compact embeddings** for genuine identities
                - Stabilizes representation of bona-fide pairs
                """)

            # ---------------- RIGHT COLUMN ----------------
            with col2:
                st.markdown("#### How it works:")
                st.markdown("""
                - ID image and selfie are passed through the **same encoder**
                - Each head learns a **different task**
                - Reduces conflict between identity recognition and morph detection
                """)

                st.markdown("#### ✅ Key Benefit")
                st.markdown(
                    "Produces more **stable**, **interpretable**, and **discriminative embeddings**."
                )

        # -------------------------------
        # TAB 3: Angular Embedding + Cosine Similarity
        # -------------------------------
        with tab3:
            st.markdown(
                "The model uses **angular embeddings inspired by ArcFace** together with "
                "**cosine similarity** for face pair comparison."
            )

            col1, col2 = st.columns(2)

            # ---------------- LEFT COLUMN ----------------
            with col1:
                st.markdown("#### Why Angular Embedding is used:")
                st.markdown("""
                - Encourages **clear separation** between identities in angular space  
                - Produces **compact clusters** for bona fide identities  
                - Pushes morph embeddings away from genuine identity clusters  
                """)

                st.markdown("#### Why Cosine Similarity is used:")
                st.markdown("""
                - Measures similarity based on **angular distance**  
                - **Stable and scale-independent**  
                - Suitable for **threshold-based decisions**  
                """)

            # ---------------- RIGHT COLUMN ----------------
            with col2:
                st.markdown("#### Decision Logic:")
                st.markdown("""
                - **High cosine similarity** → Bona fide pair  
                - **Low cosine similarity** → Morph attack  
                """)

                st.markdown("#### ✅ Key Benefit")
                st.markdown(
                    "Enables **robust and consistent morph detection** across different datasets."
                )



            

            

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
        
        
        st.markdown("""
            <style>
            /* Centers images globally within their containers */
            [data-testid="stImage"] {
                display: flex;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)

        def resize_fixed(img, size=(300, 400)):
            """Resize image to fixed width & height"""
            return img.resize(size, Image.Resampling.LANCZOS)

        img_id = Image.open(uploaded_id).convert("RGB")
        img_selfie = Image.open(uploaded_selfie).convert("RGB")

        # Force same size
        img_id_resized = resize_fixed(img_id, size=(300, 400))
        img_selfie_resized = resize_fixed(img_selfie, size=(300, 400))

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<h4>🆔 ID Document</h4>", unsafe_allow_html=True)
            st.image(img_id_resized, caption="Original ID Image")

        with col4:
            st.markdown("<h4>🤳 Live Selfie</h4>", unsafe_allow_html=True)
            st.image(img_selfie_resized, caption="Original Selfie Image")
        # -------------------------------
        # 🔄 Preprocess
        # -------------------------------
        t_id = test_transform(img_id_resized).unsqueeze(0).to(device)
        t_selfie = test_transform(img_selfie_resized).unsqueeze(0).to(device)

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
        DMAD_THRESHOLD = 0.2496
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
            
    
        if dmad_rejected or smad_rejected:
            st.subheader("🔥 Grad-CAM Visualization")
            t_id_cam = t_id.clone().detach().requires_grad_(True)
            t_selfie_cam = t_selfie.clone().detach().requires_grad_(True)
            
            st.write("S-MAD Analysis of Suspicious Regions")
            
            cam_id = grad_cam.generate(t_id_cam, class_idx=1)
            
            # 2. Create Overlay (Numpy array)
            ov_id_np = make_overlay(cam_id, t_id_cam, img_id)

            # 3. Convert Numpy to PIL & Resize (Fixes the TypeError)
            # This ensures it matches the (300, 400) size of img_id_resized
            ov_id_pil = Image.fromarray((ov_id_np * 255).astype(np.uint8))
            overlay_id_final = resize_fixed(ov_id_pil, size=(300, 400))

            # --- Grad-CAM Display (Small & Centered) ---
            # Using 1.5 spacers to keep the images tight in the middle
            sp3, c3, c4, sp4 = st.columns([1.5, 2, 2, 1.5])
            
            with c3:
                # Original Image (Already resized to 300, 400 earlier in your code)
                st.image(img_id_resized, caption="Original ID Image", use_container_width=False)
                
            with c4:
                # New resized overlay
                st.image(overlay_id_final, caption="ID Grad-CAM Analysis", use_container_width=False)
                
            st.write("D-MAD Analysis of Suspicious Regions")
            # --- Grad-CAM Processing ---
           

            # 1. Generate Raw Heatmaps (Numpy arrays)
            cam_id = grad_cam_dmad.generate(t_id_cam, t_selfie_cam, class_idx=1)
            cam_selfie = grad_cam_dmad.generate(t_selfie_cam, t_id_cam, class_idx=1)

            # 2. Create Overlays (Numpy arrays)
            ov_id_np = make_overlay(cam_id, t_id_cam, img_id)
            ov_selfie_np = make_overlay(cam_selfie, t_selfie_cam, img_selfie)

            # 3. Convert Numpy to PIL & Resize (This fixes your TypeError)
            # We wrap the numpy array in Image.fromarray so resize_fixed can work
            ov_id_pil = Image.fromarray((ov_id_np * 255).astype(np.uint8))
            ov_selfie_pil = Image.fromarray((ov_selfie_np * 255).astype(np.uint8))
            
            overlay_id_final = resize_fixed(ov_id_pil, size=(300, 400))
            overlay_selfie_final = resize_fixed(ov_selfie_pil, size=(300, 400))

            # --- Grad-CAM Display (Small & Centered) ---
            
            # Row 1: ID Comparison
            sp1, col1, col2, sp2 = st.columns([1.5, 2, 2, 1.5])
            with col1:
                st.image(img_id_resized, caption="Original ID Image", use_container_width=False)
            with col2:
                st.image(overlay_id_final, caption="ID Features", use_container_width=False)

            # Row 2: Selfie Comparison
            sp3, col3, col4, sp4 = st.columns([1.5, 2, 2, 1.5])
            with col3:
                st.image(img_selfie_resized, caption="Original Selfie Image", use_container_width=False)
            with col4:
                st.image(overlay_selfie_final, caption="Selfie Features", use_container_width=False)


        else:
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

    
    # In your UI logic:
    if uploaded_id and uploaded_selfie:

        img_id = Image.open(uploaded_id).convert("RGB")
        img_selfie = Image.open(uploaded_selfie).convert("RGB")
        
        t_id = test_transform(img_id).unsqueeze(0).to(device)
        t_selfie = test_transform(img_selfie).unsqueeze(0).to(device)

        # 1. Calculate similarity
        with torch.no_grad():
            _, cos_sim = dmad_model(t_id, t_selfie)
        cos_val, pred = cosine_decision(cos_sim, 0.2496)

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
        spacer1, col1, col2, spacer2 = st.columns([1, 2, 2, 1]) # Makes them smaller
        
        sp1, col1, col2, sp2 = st.columns([1, 2, 2, 1])
        with col1:
            st.image(img_id, caption="Original ID Image", use_container_width=True)
        with col2:
            st.image(make_overlay(cam_id, t_id_cam, img_id), caption="ID Features")

        # Bottom Row: Selfie Comparison
        sp3, col3, col4, sp4 = st.columns([1, 2, 2, 1])
        with col3:
            st.image(img_selfie, caption="Original Selfie Image", use_container_width=True)
        with col4:
            st.image(make_overlay(cam_selfie, t_selfie_cam, img_selfie), caption="Selfie Features")

                