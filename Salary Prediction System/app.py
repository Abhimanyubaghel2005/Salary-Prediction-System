import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("../salary_model.pkl", "rb"))
le_department = pickle.load(open("../dept_encoder.pkl", "rb"))
le_job = pickle.load(open("../job_encoder.pkl", "rb"))

st.title("💰 Salary Prediction System")

department = st.selectbox("Select Department", le_department.classes_)
years = st.number_input("Years of Experience", min_value=0, max_value=40)
job_rate = st.selectbox("Select Job Rate", le_job.classes_)

if st.button("Predict Salary"):

    dept_encoded = le_department.transform([department])[0]
    job_encoded = le_job.transform([job_rate])[0]

    prediction = model.predict([[dept_encoded, years, job_encoded]])


    st.success(f"Predicted Monthly Salary: ₹ {round(prediction[0], 2)}")

