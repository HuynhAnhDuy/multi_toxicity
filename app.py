import streamlit as st
from model_utils import load_models, predict_all

# Cấu hình giao diện
st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("☣️ Multi-endpoint Toxicity Predictor")

# Tên đầy đủ của 8 endpoint
ENDPOINT_NAMES = [
    "Mutagenicity",
    "Carcinogenicity",
    "hERG Inhibition",
    "Hepatotoxicity",
    "Skin Sensitization",
    "Respiratory Toxicity",
    "Developmental Toxicity",
    "Neurotoxicity"
]

# Load mô hình & file descriptor XML
models, features = load_models("models", "features")

# Nhập SMILES
st.markdown("### 🔬 Enter a SMILES string for prediction:")
smiles = st.text_input("SMILES")

if st.button("Predict") and smiles:
    try:
        results = predict_all(smiles, models, features)
        st.subheader("Prediction Results")
        for name, (label, prob) in zip(ENDPOINT_NAMES, results):
            st.markdown(f"**{name}:** {label} _(probability = {prob})_")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
