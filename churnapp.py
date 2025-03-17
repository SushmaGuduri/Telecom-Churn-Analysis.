import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("finalized_model.sav", "rb") as file:
    model = pickle.load(file)

# Apply custom styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f4f4;
            color: #333333;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("📱 Customer Churn Prediction App")

# Layout for image and inputs
col_img, col_inputs = st.columns([1, 3])

with col_img:
    st.image("C:/Users/sushm/Downloads/Churn/customerchurn.png", caption="Customer Churn Analysis", use_container_width=True)


# User Input (Three Column Layout)
col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State", [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", 
        "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", 
        "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ]) 
    account_length = st.number_input("Account Length (days)", min_value=0, max_value=500, value=100)
    voice_plan = st.radio("Has Voice Plan?", ["No", "Yes"])
    voice_messages = st.number_input("Number of Voice Messages", min_value=0, value=10)
    intl_plan = st.radio("Has International Plan?", ["No", "Yes"])
    intl_mins = st.number_input("International Minutes", min_value=0.0, value=10.0)

with col2:
    intl_calls = st.number_input("International Calls", min_value=0, value=3)
    intl_charge = st.number_input("International Charge", min_value=0.0, value=2.7)
    day_mins = st.number_input("Day Minutes", min_value=0.0, value=100.0)
    day_calls = st.number_input("Day Calls", min_value=0, value=100)
    day_charge = st.number_input("Day Charge", min_value=0.0, value=20.0)
    eve_mins = st.number_input("Evening Minutes", min_value=0.0, value=200.0)

with col3:
    eve_calls = st.number_input("Evening Calls", min_value=0, value=100)
    eve_charge = st.number_input("Evening Charge", min_value=0.0, value=15.0)
    night_mins = st.number_input("Night Minutes", min_value=0.0, value=250.0)
    night_calls = st.number_input("Night Calls", min_value=0, value=100)
    night_charge = st.number_input("Night Charge", min_value=0.0, value=10.0)
    customer_calls = st.number_input("Customer Service Calls", min_value=0, value=2)

# Encoding categorical variables
state_dict = {state: idx for idx, state in enumerate(["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", 
    "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", 
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"])}
state_encoded = state_dict[state]
voice_plan = 1 if voice_plan == "Yes" else 0
intl_plan = 1 if intl_plan == "Yes" else 0

# Prepare input data for prediction
input_data = pd.DataFrame([[state_encoded, account_length, voice_plan, voice_messages, intl_plan, intl_mins, intl_calls,
                            intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge,
                            night_mins, night_calls, night_charge, customer_calls]],
                          columns=['state', 'account.length', 'voice.plan', 'voice.messages', 'intl.plan',
                                   'intl.mins', 'intl.calls', 'intl.charge', 'day.mins', 'day.calls',
                                   'day.charge', 'eve.mins', 'eve.calls', 'eve.charge', 'night.mins',
                                   'night.calls', 'night.charge', 'customer.calls'])

# Define prediction threshold
threshold = 0.7  # Example threshold

# Prediction
if st.button("Predict Churn"):
    proba = model.predict_proba(input_data)[0][1]  # Probability of churn
    
    
    # Check threshold
    if proba > threshold:
        st.error("🚨 The customer is likely to churn.")
    else:
        st.success("✅ The customer is NOT likely to churn.")

