import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.markdown("---")


@st.cache_data
def load_results():
    return pd.read_csv("model_results.csv")


results_df = load_results()

st.subheader("📊 Model Comparison")

st.dataframe(
    results_df,
    use_container_width=True
)

st.markdown("---")

st.subheader("🏆 Best Model")

best_model = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", best_model["Model"])
col2.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
col3.metric("Recall", f"{best_model['Recall']:.4f}")
col4.metric("F1 Score", f"{best_model['F1 Score']:.4f}")

st.markdown("---")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.subheader("📈 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Models")
    ax.set_title("Accuracy Comparison")

    plt.xticks(rotation=15)

    st.pyplot(fig)

with chart_col2:

    st.subheader("📈 Recall Comparison")

    fig2, ax2 = plt.subplots(figsize=(5, 3))

    ax2.bar(
        results_df["Model"],
        results_df["Recall"]
    )

    ax2.set_ylabel("Recall")
    ax2.set_xlabel("Models")
    ax2.set_title("Recall Comparison")

    plt.xticks(rotation=15)

    st.pyplot(fig2)

st.markdown("---")

st.subheader("🔍 Fraud Prediction Using All Models")

models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

scaler = joblib.load("models/scaler.pkl")

feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]

if "feature_values" not in st.session_state:
    st.session_state.feature_values = [0.0] * 30

st.markdown("### 📋 Paste Transaction Data")

st.caption(
    "Example: 0,-1.35,-0.07,2.53,...,149.62"
)

input_text = st.text_area(
    "Transaction Input",
    height=120
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "📥 Load Data",
        use_container_width=True
    ):

        try:

            cleaned_input = input_text.replace(
                "\n",
                ","
            )

            values = [
                float(x.strip())
                for x in cleaned_input.split(",")
                if x.strip() != ""
            ]

            if len(values) != 30:

                st.error(
                    f"Expected 30 values, got {len(values)}"
                )

            else:

                st.session_state.feature_values = values

                st.success(
                    "Data loaded successfully!"
                )

        except Exception as e:

            st.error(
                f"Invalid input format: {e}"
            )

with col2:

    if st.button(
        "🗑 Clear Data",
        use_container_width=True
    ):

        st.session_state.feature_values = [0.0] * 30
        st.rerun()

st.markdown("---")

st.subheader(
    "✏️ Editable Transaction Features"
)

editor_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Value": st.session_state.feature_values
    }
)

edited_df = st.data_editor(
    editor_df,
    hide_index=True,
    use_container_width=True,
    disabled=["Feature"]
)

st.markdown("---")

predict_clicked = st.button(
    "🚀 Predict Using All Models",
    use_container_width=True
)

if predict_clicked:

    try:

        values = edited_df["Value"].tolist()

        if len(values) != 30:

            st.error(
                "Exactly 30 values required."
            )

            st.stop()

        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        scaled_data = scaler.transform(
            input_df
        )

        st.subheader(
            "📋 Prediction Results"
        )

        prediction_results = []

        fraud_votes = 0

        for name, model in models.items():

            prediction = model.predict(
                scaled_data
            )[0]

            if prediction == 1:
                fraud_votes += 1

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = (
                    model.predict_proba(
                        scaled_data
                    )[0][1]
                )

                confidence = (
                    probability * 100
                    if prediction == 1
                    else (1 - probability) * 100
                )

                confidence_text = (
                    f"{confidence:.2f}%"
                )

            else:

                confidence_text = (
                    "Not Available"
                )

            prediction_results.append(
                {
                    "Model": name,
                    "Prediction":
                        "Fraud Detected"
                        if prediction == 1
                        else "Legitimate Transaction",
                    "Confidence":
                        confidence_text
                }
            )

        prediction_df = pd.DataFrame(
            prediction_results
        )

        styled_df = prediction_df.style.map(
            lambda x:
                (
                    "background-color:#ff4b4b;color:white;"
                )
                if x == "Fraud Detected"
                else (
                    "background-color:#28a745;color:white;"
                    if x == "Legitimate Transaction"
                    else ""
                ),
            subset=["Prediction"]
        )

        st.dataframe(
            styled_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader(
            "🗳 Ensemble Voting Result"
        )

        if fraud_votes >= 3:

            st.error(
                f"⚠️ Fraud Detected ({fraud_votes}/5 models agree)"
            )

        else:

            st.success(
                f"✅ Legitimate Transaction ({5 - fraud_votes}/5 models agree)"
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )