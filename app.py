import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
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

def main():
    st.title("Customer Check-in Prediction")
    st.write("This app predicts whether a customer will check in based on their details.")

    # Load trained model and scalers
    model = tf.keras.models.load_model('my_model.keras')
    minmax_scaler = joblib.load('scaler.pkl')
    standard_scaler = joblib.load('standard_scaler.pkl')
    encoder = joblib.load('encoder.pkl')

    # Input form
    st.header("Customer Details")

    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    persons_nights = st.number_input("Persons Nights", min_value=0, max_value=365, value=0)
    avg_lead_time = st.number_input("Average Lead Time", min_value=0, max_value=365, value=0)
    lodging_revenue = st.number_input("Lodging Revenue", min_value=0.0, max_value=1000000.0, value=0.0)
    other_revenue = st.number_input("Other Revenue", min_value=0.0, max_value=100000.0, value=0.0)
    days_since_last_stay = st.number_input("Days Since Last Stay", min_value=0, max_value=5000, value=0)

    distribution_channel = st.selectbox(
        "Distribution Channel", options=encoder.categories_[0]
    )

    # Process input data
    features = np.array([
        age,
        persons_nights,
        avg_lead_time,
        lodging_revenue,
        other_revenue,
        days_since_last_stay
    ]).reshape(1, -1)

    # Scale numerical features
    features = minmax_scaler.transform(features)

    # Encode categorical features
    encoded_channel = encoder.transform([[distribution_channel]])

    # Combine all features
    features = np.hstack([features, encoded_channel])

    # Standardize features
    features = standard_scaler.transform(features)

    # Make prediction
    if st.button("Predict"):
        prediction = (model.predict(features) > 0.5).astype(int)
        result = "The customer is likely to check in." if prediction[0, 0] == 1 else "The customer is unlikely to check in."
        st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()
