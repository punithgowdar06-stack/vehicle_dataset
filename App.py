import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Encoders (cached) ---
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('xgbr_model.joblib')
    encs = joblib.load('label_encoders.joblib')

    # Ensure encoders is a dict (common-saved format). If it's not, try to convert.
    if not isinstance(encs, dict):
        try:
            encs = dict(encs)
        except Exception:
            st.error("label_encoders.joblib must be a dict mapping column names to encoders.")
            raise

    # Decode bytes in classes_ (if any) to strings so selectbox and transform use the same types
    for k, le in encs.items():
        try:
            classes = le.classes_
            if len(classes) > 0 and isinstance(classes[0], (bytes, bytearray)):
                decoded = [c.decode('utf-8') for c in classes]
                le.classes_ = np.array(decoded, dtype=object)
                encs[k] = le
        except Exception:
            # if anything goes wrong for one encoder, continue — will be caught later if needed
            pass

    # Case-insensitive mapping from expected column name -> actual encoder key
    key_map = {k.lower(): k for k in encs.keys()}
    return model, encs, key_map

model, encoders, encoder_key_map = load_model_and_encoders()

# --- App UI ---
st.title('🚗 Vehicle Price Prediction')
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")
st.sidebar.header('Vehicle Features')

def get_classes_for(col_name):
    """Return (classes_list, actual_encoder_key) for a given column name (case-insensitive)."""
    key = encoder_key_map.get(col_name.lower())
    if key:
        classes = list(encoders[key].classes_)
        return classes, key
    return None, None

def user_input_features():
    year = st.sidebar.slider('Year', 1990, 2025, 2015)
    make_classes, _ = get_classes_for('make')
    if make_classes:
        make = st.sidebar.selectbox('Make', make_classes)
    else:
        make = st.sidebar.text_input('Make')

    model_classes, _ = get_classes_for('model')
    if model_classes:
        model_input = st.sidebar.selectbox('Model', model_classes)
    else:
        model_input = st.sidebar.text_input('Model')

    trim_classes, _ = get_classes_for('trim')
    trim = st.sidebar.selectbox('Trim', trim_classes) if trim_classes else st.sidebar.text_input('Trim')

    body_classes, _ = get_classes_for('body')
    body = st.sidebar.selectbox('Body Type', body_classes) if body_classes else st.sidebar.text_input('Body Type')

    transmission_classes, _ = get_classes_for('transmission')
    transmission = st.sidebar.selectbox('Transmission', transmission_classes) if transmission_classes else st.sidebar.text_input('Transmission')

    state_classes, _ = get_classes_for('state')
    state = st.sidebar.selectbox('State', state_classes) if state_classes else st.sidebar.text_input('State')

    condition = st.sidebar.slider('Condition (1-5)', 1.0, 5.0, 3.5, 0.1)
    odometer = st.sidebar.number_input('Odometer (miles)', min_value=0, max_value=500000, value=50000)

    color_classes, _ = get_classes_for('color')
    color = st.sidebar.selectbox('Color', color_classes) if color_classes else st.sidebar.text_input('Color')

    interior_classes, _ = get_classes_for('interior')
    interior = st.sidebar.selectbox('Interior Color', interior_classes) if interior_classes else st.sidebar.text_input('Interior Color')

    seller_classes, _ = get_classes_for('seller')
    seller = st.sidebar.selectbox('Seller', seller_classes) if seller_classes else st.sidebar.text_input('Seller')

    mmr = st.sidebar.number_input('Manheim Market Report (MMR)', min_value=0, value=20000)

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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# --- Encoding (only for cols we have encoders for) ---
encoded_df = input_df.copy()

for col in input_df.columns:
    lookup_key = encoder_key_map.get(col.lower())
    if lookup_key:
        le = encoders[lookup_key]
        try:
            # Convert to string to match decoded classes, then transform
            encoded_df[col] = le.transform(encoded_df[col].astype(str))
        except Exception as e:
            st.warning(f"Encoding '{col}' failed: {e}. Falling back to most frequent class or -1.")
            try:
                encoded_df[col] = le.transform([le.classes_[0]])[0]
            except Exception:
                encoded_df[col] = -1
    else:
        # no encoder for this column (numeric columns like year, odometer, mmr)
        pass

# Convert all to numeric floats (model usually expects numeric input)
try:
    final_X = encoded_df.astype(float)
except Exception:
    # if some columns couldn't convert to float, show debug info
    st.error("Could not convert features to numeric. Here's the encoded dataframe for debugging:")
    st.write(encoded_df)
    st.stop()

# --- Prediction ---
if st.button('Predict Price'):
    try:
        prediction = model.predict(final_X)
        price_str = f'${prediction[0]:,.2f}'
        st.success(f'The estimated selling price of the vehicle is: **{price_str}**')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(final_X)
