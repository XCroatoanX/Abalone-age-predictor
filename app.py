from pathlib import Path
import sqlite3
import streamlit as st
import pandas as pd
import joblib


DB_PATH = Path("predictions.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            sex TEXT NOT NULL,
            length REAL NOT NULL,
            shell_weight REAL NOT NULL,
            height REAL NOT NULL,
            predicted_age REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(sex: str, length: float, shell_weight: float, height: float, predicted_age: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO predictions (sex, length, shell_weight, height, predicted_age)
        VALUES (?, ?, ?, ?, ?)
        """,
        (sex, length, shell_weight, height, predicted_age),
    )
    conn.commit()
    conn.close()


def load_predictions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query(
        """
        SELECT id, created_at, sex, length, shell_weight, height, predicted_age
        FROM predictions
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df_history


st.set_page_config(page_title="ML Predictor", layout="centered")
st.title("ML Predictor for Abalone Age")

init_db()

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
    selected_sex = sex_map[x4]
    input_df = pd.DataFrame({
        "Length": [x1],
        "Shell_weight": [x2],
        "Height": [x3],
        "Sex": [selected_sex],
    })

    prediction = model.predict(input_df)
    predicted_age = float(prediction[0])
    save_prediction(
        sex=selected_sex,
        length=float(x1),
        shell_weight=float(x2),
        height=float(x3),
        predicted_age=predicted_age,
    )

    st.subheader("Prediction")
    st.write(f"Predicted Age: {predicted_age:.2f} years")


st.markdown("---")
st.subheader("Prediction History")

history_df = load_predictions()

if history_df.empty:
    st.info("No predictions saved yet. Make a prediction to start tracking.")
else:
    history_df["created_at"] = pd.to_datetime(history_df["created_at"])
    st.dataframe(history_df, use_container_width=True)

    chart_df = history_df.sort_values("created_at")[["created_at", "predicted_age"]].set_index("created_at")
    st.line_chart(chart_df)
    st.caption(f"Total stored predictions: {len(history_df)}")
  