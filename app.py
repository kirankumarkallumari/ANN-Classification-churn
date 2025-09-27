import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load the trained ANN model
model = tf.keras.models.load_model('model.keras')



# Load the encoders and scaler
with open('gender_encoder.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('geo_encoder.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer fucking Churn Prediction")
st.sidebar.header("Input Customer Data")

credit_score = st.sidebar.number_input("Credit Score", min_value=0, max_value=1000, value=600)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.sidebar.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=60000.0)
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Create a DataFrame from user input
input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}
input_df = pd.DataFrame([input_data])

# Preprocess 'Gender'
gender_encoded = le_gender.transform(input_df["Gender"])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=["Gender"], index=input_df.index)

# Preprocess 'Geography'
geo_encoded = ohe_geo.transform(input_df[["Geography"]])
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=ohe_geo.get_feature_names_out(["Geography"]), 
    index=input_df.index
)

# Drop original categorical cols
input_df = input_df.drop(["Geography", "Gender"], axis=1)

# Concatenate everything
input_df = pd.concat([input_df, geo_encoded_df, gender_encoded_df], axis=1)

# Match training column order (adjust if needed)
original_columns = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain"
]
input_df = input_df[original_columns]

# Scale the input
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")
    st.write(f"Predicted Churn Probability: {prediction_proba:.4f}")
    if prediction_proba > 0.5:
        st.write("⚠️ The customer is likely to churn.")
    else:
        st.write("✅ The customer is not likely to churn.")
