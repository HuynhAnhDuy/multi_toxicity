import streamlit as st
import pandas as pd
from predict_rf_pubchem_hepa import predict_from_smiles

st.set_page_config(page_title="Toxicity Predictor", layout="centered")

st.title("🧪 Hepatotoxicity Predictor")
st.markdown("Nhập một hoặc nhiều SMILES để dự đoán liệu hợp chất có **độc tính trên gan (hepatotoxic)** hay không.")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    1. Paste SMILES hoặc upload file CSV chứa SMILES.  
    2. Nhấn **Predict** để xem kết quả.  
    """)
    st.markdown("---")
    st.markdown("""
    🔍 **Quy tắc phân loại:**  
    - **Xác suất > 0.5** → ☣️ **Toxic**  
    - **Xác suất ≤ 0.5** → ✅ **Non-toxic**
    """)
    st.markdown("---")
    st.info("Hiện tại chỉ hỗ trợ nhập SMILES trực tiếp.")

# ==== Nhập SMILES trực tiếp ====
smiles_input = st.text_area(
    "Nhập SMILES (mỗi dòng một SMILES):",
    height=200,
    placeholder="Ví dụ:\nCCO\nc1ccccc1C(=O)O\nCN(C)C=O"
)

if st.button("🔍 Dự đoán"):
    if not smiles_input.strip():
        st.warning("⚠️ Vui lòng nhập ít nhất một SMILES.")
    else:
        smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]
        
        try:
            results = predict_from_smiles(smiles_list)
            results.index += 1  # start from 1
            st.success(f"✅ Dự đoán hoàn tất cho {len(results)} hợp chất.")
            st.dataframe(results, use_container_width=True)

            # Cho phép tải kết quả
            csv = results.to_csv(index=False)
            st.download_button("📥 Tải kết quả CSV", csv, file_name="prediction_results.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
