import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Load pre-trained model and pre-processing artifacts
@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model('my_model.keras')
    scaler = joblib.load('scaler.pkl')
    standard_scaler = joblib.load('standard_scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    return model, scaler, standard_scaler, encoder

model, scaler, standard_scaler, encoder = load_model_and_artifacts()

# Streamlit app
st.title("Customer Check-In Prediction")
st.write("This application predicts whether a customer will check in based on the provided details.")

# User input form
with st.form("input_form"):
    st.subheader("Enter Customer Details")
    
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    persons_nights = st.number_input("PersonsNights", min_value=1, value=1)
    avg_lead_time = st.number_input("AverageLeadTime", min_value=0, value=30)
    lodging_revenue = st.number_input("LodgingRevenue", min_value=0.0, value=0.0, format="%.2f")
    other_revenue = st.number_input("OtherRevenue", min_value=0.0, value=0.0, format="%.2f")
    days_since_last_stay = st.number_input("DaysSinceLastStay", min_value=0, value=0)
    distribution_channel = st.selectbox("Distribution Channel", ['Direct', 'Corporate', 'Travel Agent/Operator', 'Other'])

    submit_button = st.form_submit_button("Predict")

# Process user input
if submit_button:
    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        'Age': [age],
        'PersonsNights': [persons_nights],
        'AverageLeadTime': [avg_lead_time],
        'LodgingRevenue': [lodging_revenue],
        'OtherRevenue': [other_revenue],
        'DaysSinceLastStay': [days_since_last_stay],
        'DistributionChannel': [distribution_channel]
    })

    # Add derived features
    input_data['RevenuePerNight'] = (
        (input_data['LodgingRevenue'] + input_data['OtherRevenue']) / input_data['PersonsNights']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Scale numerical features
    numerical_columns = ['Age', 'PersonsNights', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'DaysSinceLastStay', 'RevenuePerNight']
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # One-hot encode categorical features
    encoded_features = encoder.transform(input_data[['DistributionChannel']])
    encoded_features_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(['DistributionChannel']),
        index=input_data.index
    )

    # Combine scaled numerical and encoded categorical features
    input_data = pd.concat([input_data.drop(columns=['DistributionChannel']), encoded_features_df], axis=1)

    # Standardize the data
    input_data = standard_scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_label = (prediction > 0.5).astype(int)[0][0]

    # Display result
    if prediction_label == 1:
        st.success("The customer is likely to check in.")
    else:
        st.warning("The customer is unlikely to check in.")
