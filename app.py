import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import joblib
import base64

def set_background(png_file):  #Custom function to add a background image to the app. The image file
    with open(png_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:hotelre.png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('hotelre.png') #is read, encoded with base64, and applied using HTML and CSS styling.


# Load resources (model, scaler, and encoder)
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('my_model.keras')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading resources: {e}")
        st.stop()


# Preprocess input data
def preprocess_input(data, scaler, encoder):
    try:
        # Derived feature
        data['RevenuePerNight'] = (
            (data['LodgingRevenue'] + data['OtherRevenue']) / data['PersonsNights']
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Scale numerical features
        numerical_columns = [
            'Age', 'PersonsNights', 'AverageLeadTime', 'LodgingRevenue', 
            'OtherRevenue', 'DaysSinceLastStay', 'RevenuePerNight'
        ]
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        # Encode categorical features
        encoded_features = encoder.transform(data[['DistributionChannel']])
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=encoder.get_feature_names_out(['DistributionChannel']),
            index=data.index
        )

        # Combine scaled and encoded data
        data = pd.concat([data.drop(columns=['DistributionChannel']), encoded_df], axis=1)
        return data
    except Exception as e:
        st.error(f"Error preprocessing input data: {e}")
        st.stop()


# Load resources
model, scaler, encoder = load_resources()

# Streamlit app
st.title("CUSTOMER CHECK-IN PREDICTION")
st.write("This app predicts whether a customer is likely to check in based on their booking details.")

# Input form
with st.form("input_form"):
    st.subheader("Enter Customer Details")
    age = st.number_input("Age", min_value=0, max_value=100, value=30, help="Customer's age.")
    persons_nights = st.number_input("PersonsNights", min_value=1, value=1, help="Total number of person-nights")
    avg_lead_time = st.number_input("AverageLeadTime", min_value=0, value=30, help="The average time (in days) between booking creation and the check-in date.")
    lodging_revenue = st.number_input("LodgingRevenue", min_value=0.0, value=0.0, format="%.2f", help="Revenue from lodging.")
    other_revenue = st.number_input("OtherRevenue", min_value=0.0, value=0.0, format="%.2f", help="Revenue from other services.")
    days_since_last_stay = st.number_input("DaysSinceLastStay", min_value=0, value=0, help="Days since the customer's last stay.")
    distribution_channel = st.selectbox("Distribution Channel", ['Direct', 'Corporate', 'Travel Agent/Operator', 'Other'])
    submit_button = st.form_submit_button("Predict")

# Process user input and make prediction
if submit_button:
    # Create DataFrame from user input
    input_data = pd.DataFrame({
        'Age': [age],
        'PersonsNights': [persons_nights],
        'AverageLeadTime': [avg_lead_time],
        'LodgingRevenue': [lodging_revenue],
        'OtherRevenue': [other_revenue],
        'DaysSinceLastStay': [days_since_last_stay],
        'DistributionChannel': [distribution_channel]
    })

    # Preprocess the data
    processed_data = preprocess_input(input_data, scaler, encoder)

    # Display preprocessed data (optional for debugging or transparency)
    if st.checkbox("Show Preprocessed Data"):
        st.write(processed_data)

    # Make prediction
    try:
        prediction = model.predict(processed_data)
        prediction_label = (prediction > 0.5).astype(int)[0][0]

        # Display result
        if prediction_label == 1:
            st.success("The customer is likely to check in.")
        else:
            st.warning("The customer is unlikely to check in.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
