import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")

st.write(
    "Compare multiple machine learning models for predicting "
    "credit card default."
)

# ==================================================
# MODEL PATHS
# ==================================================

MODEL_PATH = "model"

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# ==================================================
# MODEL SELECTION
# ==================================================

st.subheader("Select a Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

st.write("Selected model:", selected_model)

# ==================================================
# LOAD SELECTED MODEL
# ==================================================

model_file = os.path.join(
    MODEL_PATH,
    models[selected_model]
)

model = joblib.load(model_file)

# ==================================================
# LOAD SCALER
# ==================================================

scaler_path = os.path.join(
    MODEL_PATH,
    "scaler.pkl"
)

scaler = joblib.load(scaler_path)

# ==================================================
# CSV FILE UPLOAD
# ==================================================

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

# ==================================================
# PROCESS UPLOADED FILE
# ==================================================

if uploaded_file is not None:

    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------

    data = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")

    st.write("Dataset shape:", data.shape)

    # --------------------------------------------------
    # Display Columns
    # --------------------------------------------------

    st.write("Uploaded columns:")
    st.write(list(data.columns))

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------

    st.subheader("Uploaded Dataset")

    st.dataframe(
        data.head()
    )

    # --------------------------------------------------
    # Check Target Column
    # --------------------------------------------------

    if "default" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'default' "
            "column because it is required for evaluation."
        )

        st.stop()

    # --------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------

    X_uploaded = data.drop(
        columns=["default"]
    )

    y_uploaded = data["default"]

    # --------------------------------------------------
    # Remove ID if Present
    # --------------------------------------------------

    X_uploaded = X_uploaded.drop(
        columns=["ID"],
        errors="ignore"
    )

    st.write(
        "Features used for prediction:",
        X_uploaded.shape[1]
    )

    # --------------------------------------------------
    # Scale Data
    # --------------------------------------------------

    if selected_model in [
        "Logistic Regression",
        "kNN",
        "Naive Bayes"
    ]:

        X_model = scaler.transform(
            X_uploaded
        )

    else:

        X_model = X_uploaded

    # --------------------------------------------------
    # Generate Predictions
    # --------------------------------------------------

    predictions = model.predict(
        X_model
    )

    # --------------------------------------------------
    # Generate Prediction Probabilities
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_model
    )[:, 1]

    st.success(
        "Predictions generated successfully!"
    )

    st.write(
        "Number of predictions:",
        len(predictions)
    )

    # ==================================================
    # PREDICTION RESULTS
    # ==================================================

    results = data.copy()

    results["Predicted_Default"] = predictions

    results["Default_Probability"] = probabilities

    st.subheader("Prediction Results")

    st.dataframe(
        results
    )

    # ==================================================
    # PREDICTION SUMMARY
    # ==================================================

    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Non-Default",
            int(
                (predictions == 0).sum()
            )
        )

    with col2:

        st.metric(
            "Predicted Default",
            int(
                (predictions == 1).sum()
            )
        )

    # ==================================================
    # ACTUAL VS PREDICTED
    # ==================================================

    st.subheader("Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual Default": y_uploaded,
        "Predicted Default": predictions,
        "Default Probability": probabilities
    })

    st.dataframe(
        comparison
    )

    # ==================================================
    # EVALUATION METRICS
    # ==================================================

    accuracy = (
        predictions == y_uploaded
    ).mean()

    auc = roc_auc_score(
        y_uploaded,
        probabilities
    )

    precision = precision_score(
        y_uploaded,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_uploaded,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_uploaded,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_uploaded,
        predictions
    )

    # ==================================================
    # DISPLAY EVALUATION METRICS
    # ==================================================

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with col2:

        st.metric(
            "AUC",
            f"{auc:.4f}"
        )

    with col3:

        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with col5:

        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    with col6:

        st.metric(
            "MCC",
            f"{mcc:.4f}"
        )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_uploaded,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Default",
            "Actual Default"
        ],
        columns=[
            "Predicted No Default",
            "Predicted Default"
        ]
    )

    st.dataframe(
        cm_df
    )