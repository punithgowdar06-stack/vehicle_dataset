import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Encoders ---
# Use cache to load model only once
@st.cache_resource
def load_model():
    model = joblib.load('xgbr_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    return model, encoders

model, encoders = load_model()

# --- App Title and Description ---
st.title('🚗 Vehicle Price Prediction')
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")

# --- User Input in Sidebar ---
st.sidebar.header('Vehicle Features')

def user_input_features():
    year = st.sidebar.slider('Year', 1990, 2025, 2015)
    make = st.sidebar.selectbox('Make', encoders['make'].classes_)
    model_input = st.sidebar.selectbox('Model', encoders['model'].classes_)
    trim = st.sidebar.selectbox('Trim', encoders['trim'].classes_)
    body = st.sidebar.selectbox('Body Type', encoders['body'].classes_)
    transmission = st.sidebar.selectbox('Transmission', encoders['transmission'].classes_)
    state = st.sidebar.selectbox('State', encoders['state'].classes_)
    condition = st.sidebar.slider('Condition (1-5)', 1.0, 5.0, 3.5, 0.1)
    odometer = st.sidebar.number_input('Odometer (miles)', min_value=0, max_value=500000, value=50000)
    color = st.sidebar.selectbox('Color', encoders['color'].classes_)
    interior = st.sidebar.selectbox('Interior Color', encoders['interior'].classes_)
    seller = st.sidebar.selectbox('Seller', encoders['seller'].classes_)
    mmr = st.sidebar.number_input('Manheim Market Report (MMR)', min_value=0, value=20000)

    # Create a dictionary of the inputs
    data = {
        'year': year,
        'make': make,
        'model': model_input,
        'trim': trim,
        'body': body,
        'transmission': transmission,
        'state': state,
        'condition': condition,
        'odometer': odometer,
        'color': color,
        'interior': interior,
        'seller': seller,
        'mmr': mmr,
    }
    
    # Create a DataFrame from the dictionary
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Encode User Inputs ---
# Create a copy to encode
encoded_df = input_df.copy()

# Loop through the categorical columns and apply the saved encoders
for col in encoders:
    # Use try-except to handle cases where a category might not be in the encoder
    try:
        encoded_df[col] = encoders[col].transform(encoded_df[col])
    except ValueError:
        # If a new, unseen category is entered, you might handle it by assigning a default value (e.g., -1 or mode)
        # For simplicity here, we assume inputs are from the known categories
        st.error(f"Category '{encoded_df[col].iloc[0]}' in feature '{col}' was not seen during training. Prediction may be inaccurate.")
        # As a fallback, you could assign the most frequent category's code
        encoded_df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]


# --- Prediction and Display ---
st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Price'):
    prediction = model.predict(encoded_df)
    
    st.subheader('Prediction')
    price_str = f'${prediction[0]:,.2f}'
    st.success(f'The estimated selling price of the vehicle is: **{price_str}**')
