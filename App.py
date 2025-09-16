import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Encoders (cached) ---
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('xgbr_model.joblib')
    encs = joblib.load('label_encoders.joblib')

    # Ensure encoders is a dict
    if not isinstance(encs, dict):
        encs = dict(encs)

    # Decode bytes in classes_ if needed
    for k, le in encs.items():
        if hasattr(le, "classes_"):
            if isinstance(le.classes_[0], (bytes, bytearray)):
                le.classes_ = np.array([c.decode("utf-8") for c in le.classes_], dtype=object)
    return model, encs

model, encoders = load_model_and_encoders()

# Show encoder keys for debugging
st.write("Encoders available:", list(encoders.keys()))

# --- App UI ---
st.title('🚗 Vehicle Price Prediction')
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")
st.sidebar.header('Vehicle Features')

# Known numeric features
NUMERIC_FEATURES = {
    "year": ("Year", 1990, 2025, 2015),
    "condition": ("Condition (1-5)", 1.0, 5.0, 3.5, 0.1),
    "odometer": ("Odometer (miles)", 0, 500000, 50000),
    "mmr": ("Manheim Market Report (MMR)", 0, 200000, 20000),
}

def user_input_features():
    data = {}

    # Numeric features (sliders / number inputs)
    data["year"] = st.sidebar.slider("Year", 1990, 2025, 2015)
    data["condition"] = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 3.5, 0.1)
    data["odometer"] = st.sidebar.number_input("Odometer (miles)", min_value=0, max_value=500000, value=50000)
    data["mmr"] = st.sidebar.number_input("Manheim Market Report (MMR)", min_value=0, value=20000)

    # All categorical encoders (dynamic)
    for col, le in encoders.items():
        classes = list(le.classes_)
        data[col] = st.sidebar.selectbox(col.capitalize(), classes)

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# --- Encode inputs ---
encoded_df = input_df.copy()
for col, le in encoders.items():
    try:
        encoded_df[col] = le.transform(encoded_df[col].astype(str))
    except Exception as e:
        st.warning(f"Encoding failed for {col}: {e}")
        encoded_df[col] = le.transform([le.classes_[0]])[0]

# Convert all to numeric
final_X = encoded_df.astype(float)

# --- Prediction ---
if st.button('Predict Price'):
    try:
        prediction = model.predict(final_X)
        price_str = f'${prediction[0]:,.2f}'
        st.success(f'The estimated selling price of the vehicle is: **{price_str}**')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(final_X)
