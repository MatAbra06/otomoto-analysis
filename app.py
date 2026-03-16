import streamlit as st
import pandas as pd
from joblib import load
st.write("Hello World!")

@st.cache_resource
def load_model():
    model = load('model_otomoto.pkl')
    model_columns = load('model_columns.pkl')
    return model, model_columns

car_brand = st.selectbox("Brand", ["BMW", "Audi", "Volkswagen", "Toyota", "Mercedes-Benz", "Porshe", "McLaren", "Lamborghini", "Ferrari", "Bentley"])
car_model = st.text_input("Model")
car_production_year = st.sidebar.slider("Production year", 1950, 2023)
car_mileage = st.number_input("Mileage")
car_engine_capacity = st.number_input("Engine capacity")
car_engine_power = st.number_input("Engine power")

model, model_columns = load_model()

if st.button("Quote a car"):
    car_data = {
    'vehicle_brand': car_brand,
    'vehicle_model': car_model,
    'production_year': car_production_year,
    'mileage': car_mileage,
    'engine_displacement': car_engine_capacity,
    'power': car_engine_power
    }
    df_input = pd.DataFrame(car_data, index=[0])
    df_encoded = pd.get_dummies(df_input)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    predicated_price = model.predict(df_final)
    # st.header(f"Estimated price {predicated_price[0]:,.0f} PLN")
    st.metric(label="Estimated price", value=f"{predicated_price[0]:,.0f} PLN")