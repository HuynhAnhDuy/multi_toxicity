import streamlit as st
from model_utils import load_models, predict_all

# Cấu hình giao diện
st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("☣️ Multi-endpoint Toxicity Predictor")

st.markdown("""
Predict the toxicity potential of chemical compounds using machine learning models with 8 various endpoints.

**Developers:**  
**Affiliations:** 
""", unsafe_allow_html=True)

# === Sidebar Instructions ===
with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    1. Paste a SMILES string.
    2. Click **Predict**.    
    """)

    st.markdown("---")
    st.markdown("""
    🔍 **Prediction Rule:**  
    - **Avg. probability > 0.5** → ☣️ **Possible toxicity** 
    - **Avg. probability ≤ 0.5** → ✅ **Non-toxicity**
    """)

# Tên đầy đủ của 8 endpoint
ENDPOINT_NAMES = [
    "Peripheral blood mononuclear cells toxicity",
    "Nephrotoxicity",
    "Neurotoxicity",
    "Hepatotoxicity",
    "Skin Sensitization",
    "Respiratory Toxicity",
    "Severe Cutaneous Adverse Reaction",
    "Cardiotoxicity"
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
