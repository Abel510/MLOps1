import streamlit as st
import joblib

model = joblib.load("regression.joblib")

st.title("House Price Prediction App")

size = st.number_input("Enter the size of the house (in square meters)", min_value=0.0)
nb_rooms = st.number_input("Enter the number of bedrooms", min_value=1)
garden = st.number_input("Does the house have a garden? Enter 1 for Yes, 0 for No", min_value=0, max_value=1)

if st.button("Predict"):
    input_data = [[size, nb_rooms, garden]]
    prediction = model.predict(input_data)
    st.write(f"The predicted house price is: ${prediction[0]:,.2f}")
