
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("lung_cancer_best_model.pkl")

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Lung Cancer Prediction")
st.write("Machine Learning based prediction system")

st.divider()

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

smoking = st.selectbox("Smoking", ["No", "Yes"])
yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
anxiety = st.selectbox("Anxiety", ["No", "Yes"])
peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
allergy = st.selectbox("Allergy", ["No", "Yes"])
wheezing = st.selectbox("Wheezing", ["No", "Yes"])
alcohol_consuming = st.selectbox("Alcohol Consuming", ["No", "Yes"])
coughing = st.selectbox("Coughing", ["No", "Yes"])
shortness_of_breath = st.selectbox(
    "Shortness of Breath",
    ["No", "Yes"]
)
swallowing_difficulty = st.selectbox(
    "Swallowing Difficulty",
    ["No", "Yes"]
)
chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])


def binary(value):
    return 1 if value == "Yes" else 0


if st.button("Predict", type="primary"):

    input_data = pd.DataFrame({
        "GENDER": [1 if gender == "Male" else 0],
        "AGE": [age],
        "SMOKING": [binary(smoking)],
        "YELLOW_FINGERS": [binary(yellow_fingers)],
        "ANXIETY": [binary(anxiety)],
        "PEER_PRESSURE": [binary(peer_pressure)],
        "CHRONIC_DISEASE": [binary(chronic_disease)],
        "FATIGUE": [binary(fatigue)],
        "ALLERGY": [binary(allergy)],
        "WHEEZING": [binary(wheezing)],
        "ALCOHOL_CONSUMING": [binary(alcohol_consuming)],
        "COUGHING": [binary(coughing)],
        "SHORTNESS_OF_BREATH": [binary(shortness_of_breath)],
        "SWALLOWING_DIFFICULTY": [binary(swallowing_difficulty)],
        "CHEST_PAIN": [binary(chest_pain)]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Prediction: LUNG CANCER")
    else:
        st.success("✅ Prediction: NO LUNG CANCER")

    st.metric(
        "Predicted Probability",
        f"{probability * 100:.2f}%"
    )

st.caption(
    "Educational/research project only. "
    "This tool is not a medical diagnostic system."
)
