import streamlit as st
import pandas as pd
from joblib import load
st.write("Let's predict car's price!")

@st.cache_resource
def load_model():
    model = load('model_otomoto.pkl')
    model_columns = load('model_columns.pkl')
    return model, model_columns

brand_models = {
    "BMW": ["Seria 3", "Seria 5", "X5", "M4"],
    "Audi": ["A4", "A6", "Q7", "A3"],
    "Volkswagen": ["Golf", "Passat", "Tiguan"],
    "Toyota": ["Corolla", "Yaris", "RAV4"],
    "Mercedes-Benz": ["Klasa C", "Klasa E", "GLC"],
    "Porshe": ["911", "Cayenne", "Panamera"],
    "McLaren": ["720S", "570S", "P1"],
    "Lamborghini": ["Aventador", "Huracan", "Urus"],
    "Ferrari": ["488 GTB", "Roma", "F8 Tributo"],
    "Bentley": ["Continental GT", "Bentayga"]
}

car_brand = st.selectbox("Brand", list(brand_models.keys()))
available_models = brand_models[car_brand]
car_model = st.selectbox("Model", available_models)
car_production_year = st.slider("Production year", 1950, 2023, value=2010)
car_mileage = st.number_input("Mileage", value=None, step=1, placeholder="e.g. 150000")
car_engine_capacity = st.number_input("Engine capacity", value=None, step=1, placeholder="e.g. 2000")
car_engine_power = st.number_input("Engine power", value=None, step=1, placeholder="e.g. 150")

model, model_columns = load_model()

if st.button("Quote a car"):
    if car_mileage is None or car_engine_capacity is None or car_engine_power is None:
        st.error("Please fill in all the fields before quoting!")
    else:
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
        st.metric(label="Estimated price", value=f"{predicated_price[0]:,.0f} PLN")