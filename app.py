
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# 1. Manual Mappings (To match LabelEncoder from your notebook)
# These mappings ensure the text selected in UI is converted to the correct number
workclass_map = {'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6}
marital_map = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
occupation_map = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Others': 7, 'Other-service': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
race_map = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
gender_map = {'Female': 0, 'Male': 1}

# 2. Collect ALL 13 features required by your model
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=200000)
educational_num = st.sidebar.slider("Educational Num (Years of Education)", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = 39  # Defaulting to 39 (United-States) as per standard encoding

# 3. Create DataFrame with EXACT feature names and order from training
input_df = pd.DataFrame([[
    age, 
    workclass_map[workclass], 
    fnlwgt, 
    educational_num, 
    marital_map[marital_status], 
    occupation_map[occupation], 
    relationship_map[relationship], 
    race_map[race], 
    gender_map[gender], 
    capital_gain, 
    capital_loss, 
    hours_per_week, 
    native_country
]], columns=[
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
    'capital-loss', 'hours-per-week', 'native-country'
])

st.write("### 🔎 Processed Input Data (Numeric)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    # The features now match the names and count seen at fit time
    prediction = model.predict(input_df)

    result = " earns >50K" if prediction[0] == 1 or prediction[0] == ">50K" else " earns ≤50K"
    st.success(f"✅ Prediction: The employee {result}")
