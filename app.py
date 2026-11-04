"""Streamlit evaluation app for the ML Assignment 2 classifiers.

Expected model files (saved with joblib) are placed in the ``model`` folder:
  logistic_regression.joblib
  decision_tree.joblib
  knn.joblib
  naive_bayes.joblib
  svm.joblib

Each file should contain a fitted scikit-learn estimator or Pipeline, including
any preprocessing needed to transform the raw CSV columns.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "Revenue"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}


st.set_page_config(page_title="Classifier Evaluation", page_icon="📊", layout="wide")
st.title("Online Shoppers: Classifier Evaluation")
st.write(
    "Upload a labeled test CSV, select a saved classifier, and review its "
    "performance. The app loads fitted models and does not train or refit them."
)

with st.expander("Test CSV and model file requirements", expanded=True):
    st.markdown(
        "- The CSV must contain the same input columns used during training and "
        f"the target column **`{TARGET_COLUMN}`**.\n"
        "- Save fitted scikit-learn estimators or Pipelines in the `model/` folder "
        "using the filenames listed below. A Pipeline should include all fitted "
        "preprocessing steps.\n"
        "- The model's positive class should be `True` or `1` (purchase). "
        "Probability scores are used for AUC when available."
    )

with st.sidebar:
    st.header("Evaluation settings")
    selected_name = st.selectbox("Choose a model", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")
    st.caption(f"Expected target column: `{TARGET_COLUMN}`")

if uploaded_file is None:
    st.info("Upload your test CSV using the sidebar to begin.")
    st.subheader("Expected model files")
    st.code("\n".join(f"model/{filename}" for filename in MODEL_FILES.values()))
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if test_data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if TARGET_COLUMN not in test_data.columns:
    st.error(f"The CSV must include the target column `{TARGET_COLUMN}`.")
    st.stop()

X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]
if X_test.empty:
    st.error("The CSV must include at least one input feature column.")
    st.stop()
if y_test.isna().any():
    st.error(f"The target column `{TARGET_COLUMN}` contains missing values.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_name]
if not model_path.is_file():
    st.error(f"Model file not found: `{model_path.relative_to(ROOT)}`")
    st.info("Train and save the fitted model or Pipeline at the expected path.")
    st.stop()

try:
    model = joblib.load(model_path)
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        expected_features = list(expected_features)
        missing = [column for column in expected_features if column not in X_test.columns]
        extra = [column for column in X_test.columns if column not in expected_features]
        if missing:
            st.error(f"Test CSV is missing required feature columns: {', '.join(missing)}")
            st.stop()
        if extra:
            st.warning(f"Ignoring extra CSV columns: {', '.join(extra)}")
        X_test = X_test[expected_features]

    predictions = model.predict(X_test)
except Exception as exc:
    st.error(f"Model prediction failed. Check that the CSV matches training inputs. Details: {exc}")
    st.stop()

if len(predictions) != len(y_test):
    st.error("The model prediction count does not match the number of target labels.")
    st.stop()

try:
    # UCI data represents Revenue as True/False. Normalize string booleans too.
    def normalize_label(value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return 1
            if normalized in {"false", "0", "no"}:
                return 0
        if value is True:
            return 1
        if value is False:
            return 0
        return value

    y_eval = y_test.map(normalize_label)
    pred_eval = pd.Series(predictions).map(normalize_label)
    labels = sorted(pd.unique(pd.concat([y_eval, pred_eval], ignore_index=True)), key=str)

    accuracy = accuracy_score(y_eval, pred_eval)
    precision = precision_score(y_eval, pred_eval, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0)
    recall = recall_score(y_eval, pred_eval, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0)
    f1 = f1_score(y_eval, pred_eval, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_eval, pred_eval)

    auc_value = None
    if len(labels) == 2:
        try:
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)
                classes = list(getattr(model, "classes_", []))
                positive_index = classes.index(1) if 1 in classes else classes.index(True)
                positive_scores = scores[:, positive_index]
                auc_value = roc_curve(y_eval, positive_scores, pos_label=1)
                auc_value = auc(auc_value[0], auc_value[1])
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                auc_value = roc_curve(y_eval, scores, pos_label=1)
                auc_value = auc(auc_value[0], auc_value[1])
        except (ValueError, IndexError, AttributeError):
            auc_value = None
except Exception as exc:
    st.error(f"Could not calculate metrics. Check target labels and model classes. Details: {exc}")
    st.stop()

st.subheader("Model comparison")
comparison_rows = []
comparison_errors = []
for model_name, filename in MODEL_FILES.items():
    path = MODEL_DIR / filename
    if not path.is_file():
        comparison_errors.append(f"{model_name}: missing `model/{filename}`")
        continue
    try:
        candidate = joblib.load(path)
        candidate_X = X_test.copy()
        candidate_features = getattr(candidate, "feature_names_in_", None)
        if candidate_features is not None:
            candidate_features = list(candidate_features)
            missing = [name for name in candidate_features if name not in candidate_X.columns]
            if missing:
                raise ValueError(f"test CSV is missing columns: {', '.join(missing)}")
            candidate_X = candidate_X[candidate_features]
        candidate_pred = pd.Series(candidate.predict(candidate_X)).map(normalize_label)
        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_eval, candidate_pred),
            "Precision": precision_score(y_eval, candidate_pred, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0),
            "Recall": recall_score(y_eval, candidate_pred, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0),
            "F1": f1_score(y_eval, candidate_pred, average="binary" if len(labels) == 2 else "weighted", pos_label=1, zero_division=0),
            "MCC": matthews_corrcoef(y_eval, candidate_pred),
            "AUC": None,
        }
        if len(labels) == 2 and hasattr(candidate, "predict_proba"):
            probabilities = candidate.predict_proba(candidate_X)
            classes = list(candidate.classes_)
            positive_index = classes.index(1) if 1 in classes else classes.index(True)
            row["AUC"] = roc_curve(y_eval, probabilities[:, positive_index], pos_label=1)
            row["AUC"] = auc(row["AUC"][0], row["AUC"][1])
        elif len(labels) == 2 and hasattr(candidate, "decision_function"):
            decision_scores = candidate.decision_function(candidate_X)
            row["AUC"] = roc_curve(y_eval, decision_scores, pos_label=1)
            row["AUC"] = auc(row["AUC"][0], row["AUC"][1])
        comparison_rows.append(row)
    except Exception as exc:
        comparison_errors.append(f"{model_name}: {exc}")

if comparison_rows:
    comparison = pd.DataFrame(comparison_rows).set_index("Model")
    st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
else:
    st.info("No saved models could be evaluated. Add fitted model files to the `model/` folder.")
for error in comparison_errors:
    st.warning(error)

st.subheader(f"Selected model details: {selected_name}")
st.caption(f"Evaluated {len(test_data):,} rows from the uploaded test file.")
metric_cols = st.columns(6)
metric_values = [
    ("Accuracy", accuracy),
    ("AUC", auc_value),
    ("Precision", precision),
    ("Recall", recall),
    ("F1 Score", f1),
    ("MCC", mcc),
]
for column, (label, value) in zip(metric_cols, metric_values):
    column.metric(label, "N/A" if value is None else f"{value:.4f}")
if auc_value is None:
    st.warning("AUC is unavailable for this model or test split. AUC needs probability or decision scores and both target classes.")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_eval, pred_eval, labels=labels)
    figure, axis = plt.subplots(figsize=(5, 4))
    display_labels = [str(label) for label in labels]
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with right:
    st.markdown("#### Classification report")
    report = classification_report(y_eval, pred_eval, labels=labels, target_names=display_labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview uploaded test data"):
    st.dataframe(test_data.head(20), use_container_width=True)
