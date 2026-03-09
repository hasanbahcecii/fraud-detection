import streamlit as st
import requests
import numpy as np

st.title("🕵️ Fraud Detection Web UI")
st.write("Test the FastAPI backend with synthetic fraud detection data.")

# User input: manually enter a sequence or load from file
st.write("Enter a sequence of 20 steps, each with 5 features (comma-separated).")
sample_input = st.text_area("Example (one row):", "3,5,10,50,1")

if st.button("Predict"):
    try:
        # Parse input into a list of lists
        rows = sample_input.strip().split("\n")
        sequence = [[float(x) for x in row.split(",")] for row in rows]

        payload = {"sequence": sequence}
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['class_name']} (class {result['prediction']})")
            st.json(result)
        else:
            st.error(f"API request failed! Status code: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
