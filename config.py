ENDPOINTS = {
    "Hepatotoxicity": {
        "model": "models/rf_pubchem.joblib",
        "features": "models/selected_features_pubchem.joblib",
        "xml": "descriptor_xml/PubChem.xml"
    },
    "Neurotoxicity": {
        "model": "models/rf_krfpc.joblib",
        "features": "models/selected_features_krfpc.joblib",
        "xml": "descriptor_xml/KRFPC.xml"
    },
    "Peripheral blood mononuclear cells toxicity (PBMC)": {
        "model": "models/xgb_ap2dc.joblib",
        "features": "models/selected_features_ap2dc.joblib",
        "xml": "descriptor_xml/AP2DC.xml"
    },
    "Nephrotoxicity": {
        "model": "models/xgb_krfp.joblib",
        "features": "models/selected_features_krfp.joblib",
        "xml": "descriptor_xml/KRFP.xml"
    },
    "Respiratory Toxicity": {
        "model": "models/rf_subfp.joblib",
        "features": "models/selected_features_subfp.joblib",
        "xml": "descriptor_xml/SubFP.xml"
    },
    "Severe Cutaneous Adverse Reactions (SCARs)": {
        "model": "models/rf_subfp_scar.joblib",
        "features": "models/selected_features_subfp_scar.joblib",
        "xml": "descriptor_xml/SubFP.xml"
    },
    "Cardiotoxicity": {
        "model": "models/xgb_subfpc.joblib",
        "features": "models/selected_features_subfpc.joblib",
        "xml": "descriptor_xml/SubFPC.xml"
    },
    "Skin Sensitization": {
        "model": "models/rf_pubchem_skin.joblib",
        "features": "models/selected_features_pubchem_skin.joblib",
        "xml": "descriptor_xml/PubChem.xml"
    }
}
