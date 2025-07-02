import streamlit as st
import pandas as pd
import joblib
import subprocess
import uuid
import os

from custom_preprocessing import (
    canonical_smiles,
    remove_inorganic,
    remove_mixtures
)
from config import ENDPOINTS

# ========== UI CONFIGURATION ==========
st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("☣️ Multi-endpoint Toxicity Predictor")

st.markdown("""
Predict the toxicity potential of chemical compounds using machine learning models for 8 toxicity endpoints.

**Developers:**  
**Affiliations:** 
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    1. Paste SMILES or upload CSV with SMILES.  
    2. Click **Predict**.    
    """)
    st.markdown("---")
    st.markdown("""
    🔍 **Prediction Rule:**  
    - **Probability > 0.5** → ☣️ **Toxic**  
    - **Probability ≤ 0.5** → ✅ **Non-toxic**
    """)

# ========== DESCRIPTOR CALCULATION ==========
def calculate_descriptors(smiles_list, xml_path):
    uid = str(uuid.uuid4())
    input_file = f"temp_{uid}.smi"
    output_file = f"temp_{uid}.csv"

    with open(input_file, "w") as f:
        f.write("Name,SMILES\n")
        for i, smi in enumerate(smiles_list):
            f.write(f"Mol_{i},{smi}\n")

    subprocess.run([
        "java", "-Xms2G", "-Xmx2G", "-jar", "PaDEL-Descriptor.jar",
        "-removesalt", "-standardizenitro", "-fingerprints",
        "-descriptortypes", xml_path,
        "-dir", ".", "-file", output_file, "-2d"
    ], check=True)

    df = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)
    return df

# ========== INPUT SECTION ==========
st.header("🔬 Enter Input")

input_mode = st.radio("Input SMILES via:", ["Text box", "Upload CSV"])
input_smiles = []

if input_mode == "Text box":
    text_input = st.text_area("Enter one SMILES per line")
    if text_input.strip():
        input_smiles = [s.strip() for s in text_input.strip().splitlines() if s.strip()]
else:
    uploaded_file = st.file_uploader("Upload CSV file with 'SMILES' column", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if "SMILES" in df_upload.columns:
            input_smiles = df_upload["SMILES"].dropna().astype(str).tolist()
        else:
            st.error("Uploaded file must contain a column named 'SMILES'.")

# ========== PREDICTION ==========
if st.button("Predict") and input_smiles:
    with st.spinner("🧪 Processing..."):
        # Step 1: Preprocess SMILES
        df = pd.DataFrame({"SMILES": input_smiles})
        df = canonical_smiles(df, "SMILES")
        df = remove_inorganic(df, "canonical_smiles")
        df = remove_mixtures(df, "canonical_smiles")

        if df.empty:
            st.warning("❌ No valid SMILES after preprocessing.")
            st.stop()

        smiles_clean = df["canonical_smiles"].tolist()
        results = {"SMILES": smiles_clean}

        # Step 2: Predict across all endpoints
        for endpoint_name, cfg in ENDPOINTS.items():
            st.info(f"🔄 Processing {endpoint_name}...")

            try:
                model = joblib.load(cfg["model"])
                features = joblib.load(cfg["features"])
                xml_path = cfg["xml"]

                desc_df = calculate_descriptors(smiles_clean, xml_path)
                X = desc_df.iloc[:, 1:]

                missing = set(features) - set(X.columns)
                if missing:
                    st.error(f"[{endpoint_name}] Missing features: {missing}")
                    predictions = ["Error"] * len(smiles_clean)
                    probs = [0] * len(smiles_clean)
                else:
                    X = X[features]
                    probs = model.predict_proba(X)[:, 1]
                    predictions = ["Toxic" if p > 0.5 else "Non-toxic" for p in probs]

                results[endpoint_name] = predictions
                results[f"{endpoint_name} (prob)"] = probs
            except Exception as e:
                st.error(f"[{endpoint_name}] Prediction failed: {e}")
                results[endpoint_name] = ["Error"] * len(smiles_clean)
                results[f"{endpoint_name} (prob)"] = [0] * len(smiles_clean)

        # Step 3: Show final table
        final_df = pd.DataFrame(results)
        st.success("✅ Prediction completed!")
        st.dataframe(final_df)

        st.download_button(
            label="📥 Download Results as CSV",
            data=final_df.to_csv(index=False),
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Enter SMILES and click Predict.")
