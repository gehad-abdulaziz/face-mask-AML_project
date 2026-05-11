import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Face Mask Detection")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        )
    
    if response.ok:
        result = response.json()
        if result["status"] == "mask_on":
            st.success(f"✅ {result['class']} — {result['action']}")
        else:
            st.error(f"❌ {result['class']} — {result['action']}")
        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
    else:
        st.error("API error. Is the API container running?")