import streamlit as st
import requests
import json

# MLflow Prediction Endpoint
PREDICTION_URL = "http://127.0.0.1:8000/invocations"

st.title("Car Price Prediction")
st.markdown("Fill in the car details below to predict the selling price.")

# User input form
with st.form("car_form"):
    name = st.selectbox("Car Name", ["Maruti", "Skoda", "Honda", "Hyundai", "Ambassador", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun", "Fiat", "Force", "Ford", "Jeep", "Kia", "Lexus", "Tata", "Toyota", "Volkswagen", "Volvo"])
    year = st.number_input("Manufacture Year", min_value=1990, max_value=2025, value=2014)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=120000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])
    mileage = st.text_input("Mileage (e.g. 23.4 kmpl)", "23.4 kmpl")
    engine = st.text_input("Engine (e.g. 1248 CC)", "1248 CC")
    max_power = st.text_input("Max Power (e.g. 74 bhp)", "74 bhp")
    seats = st.number_input("Seats", min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("Predict")

# Send request to MLflow prediction server
if submitted:
    input_record = {
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }

    payload = {
        "dataframe_records": [input_record]
    }

    try:
        response = requests.post(
            PREDICTION_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Predicted Selling Price: {int(prediction[0]):,} VND")
        else:
            st.error(f"Server error: {response.status_code}")
            st.code(response.text)

    except Exception as e:
        st.error("Failed to connect to the prediction server.")
        st.exception(e)
