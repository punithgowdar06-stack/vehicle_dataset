import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Encoders ---
@st.cache_resource
def load_model():
    model = joblib.load("xgbr_model.joblib")
    encoders = joblib.load("label_encoders.joblib")
    return model, encoders

model, encoders = load_model()

# --- App Title and Description ---
st.title("🚗 Vehicle Price Prediction")
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")

# --- Sidebar Input ---
st.sidebar.header("Vehicle Features")

def user_input_features():
    year = st.sidebar.slider("Year", 1990, 2025, 2015)
    make = st.sidebar.selectbox("Make", encoders["make"].classes_)
    model_input = st.sidebar.selectbox("Model", encoders["model"].classes_)
    trim = st.sidebar.selectbox("Trim", encoders["trim"].classes_)
    body = st.sidebar.selectbox("Body Type", encoders["body"].classes_)
    transmission = st.sidebar.selectbox("Transmission", encoders["transmission"].classes_)
    state = st.sidebar.selectbox("State", encoders["state"].classes_)
    condition = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 3.5, 0.1)
    odometer = st.sidebar.number_input("Odometer (miles)", min_value=0, max_value=500000, value=50000)
    color = st.sidebar.selectbox("Color", encoders["color"].classes_)
    interior = st.sidebar.selectbox("Interior Color", encoders["interior"].classes_)
    seller = st.sidebar.selectbox("Seller", encoders["seller"].classes_)
    mmr = st.sidebar.number_input("Manheim Market Report (MMR)", min_value=0, value=20000)

    data = {
        "year": year,
        "make": make,
        "model": model_input,
        "trim": trim,
        "body": body,
        "transmission": transmission,
        "state": state,
        "condition": condition,
        "odometer": odometer,
        "color": color,
        "interior": interior,
        "seller": seller,
        "mmr": mmr,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Encode User Inputs ---
encoded_df = input_df.copy()

categorical_cols = list(encoders.keys())  # only encode saved categorical features

for col in categorical_cols:
    try:
        encoded_df[col] = encoders[col].transform(encoded_df[col])
    except ValueError:
        # unseen category -> replace with most frequent category from training
        fallback = encoders[col].classes_[0]
        st.warning(f"⚠️ '{encoded_df[col].iloc[0]}' not seen in training for '{col}'. Using '{fallback}' instead.")
        encoded_df[col] = encoders[col].transform([fallback])[0]

# --- Ensure Correct Feature Order ---
expected_features = model.get_booster().feature_names  # works for XGBoost
encoded_df = encoded_df[expected_features]

# --- Prediction ---
st.subheader("User Input Features")
st.write(input_df)

if st.button("Predict Price"):
    prediction = model.predict(encoded_df)
    price_str = f"${prediction[0]:,.2f}"
    st.success(f"💰 Estimated Selling Price: **{price_str}**")
