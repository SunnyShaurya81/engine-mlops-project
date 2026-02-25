import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Engine Health Monitor", layout="wide")

st.title("🚗 Engine Predictive Maintenance Dashboard")

st.write(
    "Predict engine health using sensor data. "
    "Adjust the decision threshold to balance false alarms vs missed failures."
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()
feature_names = model.feature_names_in_

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙ Prediction Settings")

threshold = st.sidebar.slider(
    "Failure Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01,
    help="Lower → detect more faults\nHigher → reduce false alarms"
)

st.sidebar.markdown("---")
st.sidebar.write("**Engine Condition Meaning**")
st.sidebar.write("0 → Normal")
st.sidebar.write("1 → Faulty")

# -----------------------------
# SEVERITY FUNCTION
# -----------------------------
def get_severity(prob):
    if prob < 0.4:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# -----------------------------
# MANUAL PREDICTION
# -----------------------------
st.header("🔧 Manual Prediction")

cols = st.columns(3)
inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(feature, value=0.0)
        inputs.append(val)

if st.button("Predict Engine Condition"):

    input_df = pd.DataFrame([inputs], columns=feature_names)

    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= threshold else 0
    severity = get_severity(prob)

    # Result
    if prediction == 1:
        st.error("⚠ Engine Likely Faulty")
    else:
        st.success("✅ Engine Operating Normally")

    st.write(f"### Failure Probability: **{prob:.3f}**")
    st.write(f"### Severity Level: {severity}")

    # Gauge Chart
    fig, ax = plt.subplots()
    ax.barh(["Risk"], [prob])
    ax.set_xlim(0,1)
    ax.set_title("Failure Risk Level")
    st.pyplot(fig)

# -----------------------------
# BATCH PREDICTION
# -----------------------------
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if len(df) > 20000:
            st.warning("⚠ Maximum 10,000 rows allowed.")
        else:
            missing_cols = [col for col in feature_names if col not in df.columns]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                df = df[feature_names]

                probs = model.predict_proba(df)[:, 1]
                df["Failure_Probability"] = probs
                df["Prediction"] = (probs >= threshold).astype(int)
                df["Severity"] = df["Failure_Probability"].apply(get_severity)

                st.success("✅ Predictions completed")

                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results",
                    csv,
                    "engine_predictions.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# MODEL INFO
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Information")

st.write("✔ Algorithm: Random Forest")
st.write("✔ Handles feature correlation & non-linearity")
st.write("✔ Optimized for predictive maintenance")

st.info(
    "Tip: Lower threshold if missing failures is costly.\n"
    "Raise threshold if false alarms are costly."
)

st.markdown("---")
st.caption("Built for predictive maintenance monitoring")
