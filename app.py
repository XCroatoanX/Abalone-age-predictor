from pathlib import Path
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="ML Predictor", layout="centered")
st.title("ML Predictor for Abalone Age")

model_path = "abalone_age_gbr.pkl"

if not Path(model_path).exists():
    st.error(f"Model file not found: {Path(model_path).resolve()}")
    st.stop()

model = joblib.load(model_path)

st.subheader("Input Features")

x1 = st.number_input("Length (mm)", min_value=0.0, step=0.005, format="%.3f")
x2 = st.number_input("Shell weight (g)", min_value=0.0, step=0.0005, format="%.4f")
x3 = st.number_input("Height (mm)", min_value=0.0, step=0.005, format="%.3f")
x4 = st.selectbox(
    "Sex",
    options=["M (Male)", "F (Female)", "I (Infant)"],
    index=0
)

sex_map = {
    "M (Male)": "M",
    "F (Female)": "F",
    "I (Infant)": "I",
}

if st.button("Predict Age"):
    input_df = pd.DataFrame({
        "Length": [x1],
        "Shell_weight": [x2],
        "Height": [x3],
        "Sex": [sex_map[x4]],
    })

    prediction = model.predict(input_df)
    st.subheader("Prediction")
    st.write(f"Predicted Age: {prediction[0]:.2f}")
  