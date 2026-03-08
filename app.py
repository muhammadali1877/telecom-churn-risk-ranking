import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Telecom Churn Risk Ranking", page_icon="📉", layout="wide")

st.title("📉 Telecom Customer Churn Risk Ranking")
st.write("Upload a customer CSV file to generate churn probabilities, predicted churn labels, and risk segments.")

st.warning("This app is for educational and portfolio purposes.")

# =========================
# Load Saved Model Artifact
# =========================
@st.cache_resource
def load_artifact():
    artifact = joblib.load("telecom_churn_model.pkl")
    return artifact

artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]

st.info(f"Loaded model successfully. Default churn threshold = {threshold:.2f}")

# =========================
# Helper: Risk Segmentation
# =========================
def risk_segment(prob):
    if prob >= 0.70:
        return "Very High Risk"
    elif prob >= 0.50:
        return "High Risk"
    elif prob >= threshold:
        return "Medium Risk"
    else:
        return "Low Risk"

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # Required Columns Check
    # =========================
    required_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
        "CLTV"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error("The uploaded file is missing required columns:")
        st.write(missing_cols)
    else:
        # =========================
        # Feature Engineering
        # =========================
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Avg Monthly Spend"] = df["Total Charges"] / (df["Tenure Months"] + 1)

        # =========================
        # Prediction
        # =========================
        y_proba = model.predict_proba(df)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        risk_df = df.copy()
        risk_df["Churn Probability"] = y_proba
        risk_df["Predicted Churn"] = y_pred
        risk_df["Risk Segment"] = risk_df["Churn Probability"].apply(risk_segment)

        risk_df = risk_df.sort_values(by="Churn Probability", ascending=False)

        # =========================
        # Display Results
        # =========================
        st.subheader("Top 20 Highest Risk Customers")
        st.dataframe(risk_df.head(20), use_container_width=True)

        st.subheader("Risk Segment Distribution")
        st.dataframe(
            risk_df["Risk Segment"].value_counts().rename_axis("Risk Segment").reset_index(name="Count"),
            use_container_width=True
        )

        st.subheader("Predicted Churn Summary")
        churn_count = int(risk_df["Predicted Churn"].sum())
        total_count = len(risk_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_count)
        col2.metric("Predicted Churn Customers", churn_count)
        col3.metric("Predicted Churn Rate", f"{(churn_count / total_count) * 100:.2f}%")

        # =========================
        # Download Results
        # =========================
        csv_data = risk_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Risk Ranking CSV",
            data=csv_data,
            file_name="customer_churn_risk_ranking.csv",
            mime="text/csv"
        )