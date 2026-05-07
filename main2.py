import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================
st.title("🧠 Mental Health & Vulnerability Predictor")

st.write(
    "A beginner-friendly AI project made using "
    "Machine Learning and Streamlit."
)

st.caption(
    "⚠ This project is for educational purposes only "
    "and not medical advice."
)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📌 About This Project")

st.sidebar.write("""
This project predicts:
- Mental health vulnerability
- Possible mental health condition

Built using:
- Python
- Streamlit
- Scikit-learn
""")

# ==========================================
# LOAD DATA
# ==========================================
data = pd.read_csv(
    "synthetic_mental_health_lifestyle_with_condition.csv"
)

# ==========================================
# CLEAN DATA
# ==========================================
data["Mental Health Condition"] = data[
    "Mental Health Condition"
].fillna("Healthy")

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================
categorical_columns = [
    "Gender",
    "Diet",
    "Smoking Habit",
    "Mental Health History"
]

data_encoded = pd.get_dummies(
    data,
    columns=categorical_columns
)

# ==========================================
# MODEL 1 : VULNERABILITY PREDICTION
# ==========================================
X_vuln = data_encoded.drop(
    columns=["Vulnerable", "Mental Health Condition"]
)

y_vuln = data_encoded["Vulnerable"]

X_train_vuln, X_test_vuln, y_train_vuln, y_test_vuln = train_test_split(
    X_vuln,
    y_vuln,
    test_size=0.2,
    random_state=42
)

vuln_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

vuln_model.fit(
    X_train_vuln,
    y_train_vuln
)

vuln_predictions = vuln_model.predict(X_test_vuln)

vuln_accuracy = accuracy_score(
    y_test_vuln,
    vuln_predictions
)

# ==========================================
# MODEL 2 : CONDITION PREDICTION
# ==========================================
condition_data = data.copy()

condition_data = condition_data[
    condition_data["Mental Health Condition"] != "Healthy"
]

condition_encoded = pd.get_dummies(
    condition_data,
    columns=categorical_columns
)

X_cond = condition_encoded.drop(
    columns=["Mental Health Condition", "Vulnerable"]
)

y_cond = condition_encoded["Mental Health Condition"]

X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(
    X_cond,
    y_cond,
    test_size=0.2,
    random_state=42
)

cond_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

cond_model.fit(
    X_train_cond,
    y_train_cond
)

cond_predictions = cond_model.predict(X_test_cond)

cond_accuracy = accuracy_score(
    y_test_cond,
    cond_predictions
)

# ==========================================
# SAVE MODELS
# ==========================================
joblib.dump(
    vuln_model,
    "vulnerability_model.pkl"
)

joblib.dump(
    cond_model,
    "condition_model.pkl"
)

# ==========================================
# MODEL ACCURACY
# ==========================================
st.subheader("📊 Model Accuracy")

st.write(
    f"✅ Vulnerability Model Accuracy: {vuln_accuracy:.2f}"
)

st.write(
    f"✅ Condition Model Accuracy: {cond_accuracy:.2f}"
)

# ==========================================
# DATASET INSIGHTS
# ==========================================
st.subheader("📈 Dataset Insights")

st.write("Stress Level Distribution")
st.bar_chart(
    data["Stress Level"].value_counts()
)

st.write("Average Sleep Hours")
st.bar_chart(
    data["Sleep Hours"]
)

# ==========================================
# USER INPUT SECTION
# ==========================================
st.subheader("📝 Enter User Details")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=20
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

sleep_hours = st.slider(
    "Sleep Hours",
    0,
    12,
    7
)

exercise = st.slider(
    "Exercise Frequency Per Week",
    0,
    7,
    3
)

diet = st.selectbox(
    "Diet Quality",
    ["Poor", "Average", "Good"]
)

stress_level = st.slider(
    "Stress Level",
    1,
    10,
    5
)

social_interaction = st.slider(
    "Social Interaction",
    0,
    20,
    10
)

alcohol = st.slider(
    "Alcohol Consumption Per Week",
    0,
    10,
    1
)

smoking = st.selectbox(
    "Smoking Habit",
    ["Yes", "No"]
)

work_hours = st.slider(
    "Work Hours Per Week",
    0,
    100,
    40
)

mental_history = st.selectbox(
    "Mental Health History",
    ["Yes", "No"]
)

# ==========================================
# PREDICTION BUTTON
# ==========================================
if st.button("🔍 Predict"):

    with st.spinner("Analyzing mental health..."):

        input_data = pd.DataFrame({
            "Age": [age],
            "Sleep Hours": [sleep_hours],
            "Exercise Frequency": [exercise],
            "Stress Level": [stress_level],
            "Social Interaction": [social_interaction],
            "Alcohol Consumption": [alcohol],
            "Work Hours": [work_hours],
            "Gender": [gender],
            "Diet": [diet],
            "Smoking Habit": [smoking],
            "Mental Health History": [mental_history]
        })

        # Encode user input
        input_encoded = pd.get_dummies(input_data)

        # Match columns with training data
        input_encoded = input_encoded.reindex(
            columns=X_vuln.columns,
            fill_value=0
        )

        # ==========================================
        # VULNERABILITY PREDICTION
        # ==========================================
        vulnerability_prediction = vuln_model.predict(
            input_encoded
        )

        is_vulnerable = vulnerability_prediction[0] == 1

        # ==========================================
        # IF USER IS VULNERABLE
        # ==========================================
        if is_vulnerable:

            st.warning(
                "⚠ The user may be mentally vulnerable."
            )

            # ==========================================
            # CONDITION PREDICTION
            # ==========================================
            condition_prediction = cond_model.predict(
                input_encoded
            )

            predicted_condition = condition_prediction[0]

            st.error(
                f"🧠 Possible Mental Health Condition: "
                f"{predicted_condition}"
            )

            # ==========================================
            # HEALTH SUGGESTIONS
            # ==========================================
            st.subheader("💡 Helpful Suggestions")

            if stress_level > 7:
                st.info(
                    "Try stress management activities "
                    "like meditation or yoga."
                )

            if sleep_hours < 6:
                st.info(
                    "Getting better sleep may improve "
                    "mental health."
                )

            if social_interaction < 5:
                st.info(
                    "Spending time with friends or family "
                    "may help."
                )

            if exercise < 2:
                st.info(
                    "Regular exercise can improve "
                    "mental well-being."
                )

        # ==========================================
        # IF USER IS NOT VULNERABLE
        # ==========================================
        else:

            st.success(
                "✅ The user is not predicted "
                "to be vulnerable."
            )

# ==========================================
# DETAILED REPORTS
# ==========================================
with st.expander("📄 Detailed Model Reports"):

    st.write("Vulnerability Model Report")

    st.text(
        classification_report(
            y_test_vuln,
            vuln_predictions
        )
    )

    st.write("Condition Model Report")

    st.text(
        classification_report(
            y_test_cond,
            cond_predictions
        )
    )