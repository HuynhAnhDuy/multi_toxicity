import streamlit as st
# Replace this with your actual GA4 ID
GA_TRACKING_ID = "G-Y27P57QD3C"  #

GA_JS = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Y27P57QD3C"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', 'G-Y27P57QD3C');
</script>
"""

st.markdown(GA_JS, unsafe_allow_html=True)
st.markdown("✅ GA script injected!")

import pandas as pd
from PIL import Image
import sys
# === Import prediction functions ===
from predict_rf_pubchem_hepa import predict_from_smiles as predict_hepa
from predict_rf_krfpc_neu import predict_from_smiles as predict_neuro
from predict_rf_subfp_res import predict_from_smiles as predict_respo
from predict_rf_subfp_scar import predict_from_smiles as predict_scar
from predict_rf_pubchem_skin import predict_from_smiles as predict_skin
from predict_xgb_ap2dc_pbmc import predict_from_smiles as predict_pbmc
from predict_xgb_krfp_renal import predict_from_smiles as predict_renal
from predict_xgb_subfpc_car import predict_from_smiles as predict_cardio

# === Define endpoints ===
ENDPOINTS = {
    "Hepatotoxicity": predict_hepa,
    "Neurotoxicity": predict_neuro,
    "Respiratory Toxicity": predict_respo,
    "Severe Cutaneous Adverse Reactions": predict_scar,
    "Skin Sensitization": predict_skin,
    "Peripheral Blood Mononuclear Cells Toxicity": predict_pbmc,
    "Nephrotoxicity": predict_renal,
    "Cardiotoxicity": predict_cardio,
}

# === Page setup ===
st.set_page_config(page_title="Multi-endpoint Toxicity Predictor", layout="wide")
st.title("🧪 Multi-endpoint Toxicity Predictor")

# === Sidebar instructions ===
with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    Enter up to 5 SMILES (one per line).  
    Each SMILES will be evaluated across 8 toxicity endpoints.
    """)
    st.markdown("### 🔍 Prediction Rule")
    st.markdown("- Probability > 0.5 → ☣️ Toxic  \n- Probability ≤ 0.5 → ✅ Non-toxic")

# === Input ===
smiles_input = st.text_area(
    "Enter SMILES (one per line):", height=200,
    placeholder="Example:\nCCO\nc1ccccc1C(=O)O\nCN(C)C=O"
)

# === Prediction ===
if st.button("🔍 Predict"):
    if not smiles_input.strip():
        st.warning("⚠️ Please enter at least one SMILES.")
    else:
        smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]

        if len(smiles_list) > 5:
            st.error("⚠️ Please enter **no more than 5 SMILES** at a time.")
        else:
            try:
                results = []

                for smiles in smiles_list:
                    st.markdown(f"### 💊 SMILES: `{smiles}`")
                    row = []

                    for endpoint, predict_fn in ENDPOINTS.items():
                        df = predict_fn([smiles])  # pass as list
                        prob = df.loc[0, "Probability"]
                        label = df.loc[0, "Prediction"]
                        row.append((endpoint, prob, label))

                    # Show in table format
                    result_df = pd.DataFrame(row, columns=["Endpoint", "Probability", "Prediction"])
                    st.table(result_df)

                    # Save to export later
                    result_df.insert(0, "SMILES", smiles)
                    results.append(result_df)

                # Export CSV
                final_df = pd.concat(results, ignore_index=True)
                csv = final_df.to_csv(index=False)
                st.download_button("📥 Download Results as CSV", csv, "multi_toxicity_results.csv", "text/csv")
                st.success("✅ Predictions completed!")

            except Exception as e:
                st.error(f"❌ Error occurred: {e}")
# === Author Section ===
st.markdown("---")
st.subheader("👨‍🔬 About the Authors")

col1, col2, col3 = st.columns(3)

with col1:
    image1 = Image.open("assets/job.jpg")
    st.image(image1, caption="Sastiya Kampaengsri", width=160)
    st.markdown("""
    **Dr. Sastiya Kampaengsri**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction, Drug encapsulation for cancer therapy*  
    📧 [sastiya.ks@gmail.com](mailto:sastiya.ks@gmail.com)
    """)

with col2:
    image2 = Image.open("assets/duy.jpg")
    st.image(image2, caption="Huynh Anh Duy", width=160)
    st.markdown("""
    **Huynh Anh Duy**  
    Can Tho University, Vietnam  
    PhD Candidate, Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
    """)

with col3:
    image3 = Image.open("assets/tarasi.png")
    st.image(image3, caption="Tarapong Srisongkram", width=160)
    st.markdown("""
    **Asst Prof. Dr. Tarapong Srisongkram**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
    """)

# === Footer ===
st.markdown("---")
st.caption(f"🔧 Python version: {sys.version.split()[0]}")