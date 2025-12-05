import streamlit as st
import torch
from PIL import Image

from models.dmad_model import DMAD_SiameseArcFace
from models.smad_model import load_smad_model
from utils.transforms import test_transform
from utils.cosine_threshold import cosine_decision
from utils.gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
dmad_model = DMAD_SiameseArcFace(...)
dmad_model.load_state_dict(torch.load("../weights/dmad_checkpoint.pth", map_location=device)["model_state_dict"])
dmad_model.to(device)
dmad_model.eval()

smad_model = load_smad_model()
smad_model.load_state_dict(torch.load("../weights/smad_checkpoint.pth", map_location=device))
smad_model.to(device)
smad_model.eval()

st.title("Morphing Explainability System")
