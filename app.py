import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("rf_ev_fault_detector.pkl")  # or your XGBoost model filename

st.title("🚗 EV Predictive Maintenance Dashboard")
st.markdown("Upload your EV sensor data (CSV) to predict faults using the trained model.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.subheader("📋 Input Data Preview")
    st.write(df_input.head())

    features = ['voltage', 'current', 'temperature', 'vibration', 'soc']
    predictions = model.predict(df_input[features])
    df_input['Predicted Fault'] = predictions

    st.subheader("🔍 Prediction Results")
    st.write(df_input[['voltage', 'current', 'temperature', 'vibration', 'soc', 'Predicted Fault']].head())

    st.subheader("📊 Fault Prediction Summary")
    st.bar_chart(df_input['Predicted Fault'].value_counts())

    csv = df_input.to_csv(index=False).encode()
    st.download_button("Download Prediction CSV", data=csv, file_name='predictions.csv', mime='text/csv')

else:
    st.info("Upload a CSV with columns: voltage, current, temperature, vibration, soc.")
