import streamlit as st
import pandas as pd
import joblib
import numpy as np


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# Load Model
# ============================================================

model = joblib.load("lung_cancer_best_model.pkl")


# ============================================================
# Helper Function
# ============================================================

def binary(value):
    return 1 if value == "Yes" else 0


# ============================================================
# Prediction Function
# ============================================================

def predict_lung_cancer(input_data):

    # Feature order used during model training
    feature_order = model["feature_order"]

    # Arrange input exactly according to training order
    X = input_data[feature_order].copy()

    # AGE was standardized during training
    if "AGE" in X.columns:

        age_mean = float(model["age_mean"])
        age_scale = float(model["age_scale"])

        X["AGE"] = (
            X["AGE"] - age_mean
        ) / age_scale

    # Convert to NumPy
    X = X.astype(float).values

    # Logistic Regression parameters
    coef = np.asarray(
        model["coef"],
        dtype=float
    ).reshape(-1)

    intercept = float(
        np.asarray(
            model["intercept"],
            dtype=float
        ).reshape(-1)[0]
    )

    # Check dimensions
    if X.shape[1] != len(coef):

        st.error(
            f"Feature mismatch: input has "
            f"{X.shape[1]} features, but model has "
            f"{len(coef)} coefficients."
        )

        st.stop()

    # Logistic Regression equation
    z = np.dot(X[0], coef) + intercept

    # Sigmoid probability
    probability = 1 / (1 + np.exp(-z))

    # Classification threshold
    prediction = int(probability >= 0.5)

    return prediction, float(probability)


# ============================================================
# Application UI
# ============================================================

st.title("🫁 Lung Cancer Prediction")

st.write(
    "Machine learning based lung cancer prediction system."
)

st.divider()


# ============================================================
# User Inputs
# ============================================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

yellow_fingers = st.selectbox(
    "Yellow Fingers",
    ["No", "Yes"]
)

anxiety = st.selectbox(
    "Anxiety",
    ["No", "Yes"]
)

peer_pressure = st.selectbox(
    "Peer Pressure",
    ["No", "Yes"]
)

chronic_disease = st.selectbox(
    "Chronic Disease",
    ["No", "Yes"]
)

fatigue = st.selectbox(
    "Fatigue",
    ["No", "Yes"]
)

allergy = st.selectbox(
    "Allergy",
    ["No", "Yes"]
)

wheezing = st.selectbox(
    "Wheezing",
    ["No", "Yes"]
)

alcohol_consuming = st.selectbox(
    "Alcohol Consuming",
    ["No", "Yes"]
)

coughing = st.selectbox(
    "Coughing",
    ["No", "Yes"]
)

shortness_of_breath = st.selectbox(
    "Shortness of Breath",
    ["No", "Yes"]
)

swallowing_difficulty = st.selectbox(
    "Swallowing Difficulty",
    ["No", "Yes"]
)

chest_pain = st.selectbox(
    "Chest Pain",
    ["No", "Yes"]
)


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Predict",
    type="primary"
):

    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "GENDER": [
            1 if gender == "Male" else 0
        ],

        "AGE": [
            age
        ],

        "SMOKING": [
            binary(smoking)
        ],

        "YELLOW_FINGERS": [
            binary(yellow_fingers)
        ],

        "ANXIETY": [
            binary(anxiety)
        ],

        "PEER_PRESSURE": [
            binary(peer_pressure)
        ],

        "CHRONIC_DISEASE": [
            binary(chronic_disease)
        ],

        "FATIGUE": [
            binary(fatigue)
        ],

        "ALLERGY": [
            binary(allergy)
        ],

        "WHEEZING": [
            binary(wheezing)
        ],

        "ALCOHOL_CONSUMING": [
            binary(alcohol_consuming)
        ],

        "COUGHING": [
            binary(coughing)
        ],

        "SHORTNESS_OF_BREATH": [
            binary(shortness_of_breath)
        ],

        "SWALLOWING_DIFFICULTY": [
            binary(swallowing_difficulty)
        ],

        "CHEST_PAIN": [
            binary(chest_pain)
        ]
    })


    # --------------------------------------------------------
    # Make Prediction
    # --------------------------------------------------------

    prediction, probability = predict_lung_cancer(
        input_data
    )


    # ========================================================
    # Prediction Result
    # ========================================================

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: Higher Lung Cancer Risk"
        )

    else:

        st.success(
            "✅ Model Prediction: Lower Lung Cancer Risk"
        )


    # ========================================================
    # Probability
    # ========================================================

    st.metric(
        "Predicted Probability",
        f"{probability * 100:.2f}%"
    )


    # ========================================================
    # Personalized Suggestions
    # ========================================================

    st.subheader("💡 Personalized Suggestions")

    suggestions = []


    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if smoking == "Yes":

        suggestions.append(
            "🚭 Smoking is an important lung-health risk factor. "
            "Consider quitting smoking and seek professional "
            "support if needed."
        )


    # --------------------------------------------------------
    # Alcohol
    # --------------------------------------------------------

    if alcohol_consuming == "Yes":

        suggestions.append(
            "🍺 Consider reducing or avoiding alcohol consumption "
            "to support overall health."
        )


    # --------------------------------------------------------
    # Coughing
    # --------------------------------------------------------

    if coughing == "Yes":

        suggestions.append(
            "🫁 You reported coughing. If it is persistent, "
            "worsening, or unusual for you, consider discussing "
            "it with a healthcare professional."
        )


    # --------------------------------------------------------
    # Wheezing
    # --------------------------------------------------------

    if wheezing == "Yes":

        suggestions.append(
            "🫁 You reported wheezing. Persistent or worsening "
            "wheezing should be evaluated by a healthcare professional."
        )


    # --------------------------------------------------------
    # Shortness of Breath
    # --------------------------------------------------------

    if shortness_of_breath == "Yes":

        suggestions.append(
            "🫁 You reported shortness of breath. If this is "
            "persistent or worsening, consider seeking medical evaluation."
        )


    # --------------------------------------------------------
    # Chest Pain
    # --------------------------------------------------------

    if chest_pain == "Yes":

        suggestions.append(
            "❤️ You reported chest pain. Please do not rely on "
            "this ML prediction alone; discuss this symptom "
            "with a healthcare professional."
        )


    # --------------------------------------------------------
    # Swallowing Difficulty
    # --------------------------------------------------------

    if swallowing_difficulty == "Yes":

        suggestions.append(
            "🍽️ You reported swallowing difficulty. Persistent "
            "swallowing problems should be evaluated by a healthcare professional."
        )


    # --------------------------------------------------------
    # Fatigue
    # --------------------------------------------------------

    if fatigue == "Yes":

        suggestions.append(
            "😴 You reported fatigue. If it is persistent or "
            "unexplained, consider discussing it with a healthcare professional."
        )


    # --------------------------------------------------------
    # Chronic Disease
    # --------------------------------------------------------

    if chronic_disease == "Yes":

        suggestions.append(
            "🩺 You reported a chronic disease. Continue appropriate "
            "follow-up and management with your healthcare professional."
        )


    # --------------------------------------------------------
    # Multiple Symptoms
    # --------------------------------------------------------

    symptom_count = sum([
        coughing == "Yes",
        wheezing == "Yes",
        shortness_of_breath == "Yes",
        chest_pain == "Yes",
        swallowing_difficulty == "Yes",
        fatigue == "Yes"
    ])


    if symptom_count >= 3:

        suggestions.append(
            "⚠️ Several symptoms were reported. Consider getting "
            "a professional medical evaluation rather than relying "
            "on this prediction alone."
        )


    # --------------------------------------------------------
    # Model Probability Based Suggestion
    # --------------------------------------------------------

    if probability >= 0.70:

        suggestions.append(
            "📋 The model produced a relatively high predicted "
            "probability. Consider discussing your risk factors "
            "and symptoms with a qualified healthcare professional."
        )

    elif probability >= 0.40:

        suggestions.append(
            "📋 The model produced an intermediate predicted "
            "probability. Consider monitoring your symptoms and "
            "discussing relevant concerns with a healthcare professional."
        )

    else:

        suggestions.append(
            "🌱 The model produced a lower predicted probability. "
            "Continue healthy lifestyle habits and seek professional "
            "advice if you develop persistent or concerning symptoms."
        )


    # --------------------------------------------------------
    # Display Suggestions
    # --------------------------------------------------------

    for suggestion in suggestions:

        st.write(
            f"- {suggestion}"
        )


    # ========================================================
    # Disclaimer
    # ========================================================

    st.warning(
        "⚠️ Important: These suggestions are generated from "
        "the information entered into this educational ML system. "
        "The prediction is not a medical diagnosis and should "
        "not replace professional medical advice."
    )


# ============================================================
# Footer
# ============================================================

st.caption(
    "Educational/research project only. "
    "Not intended for medical diagnosis."
)