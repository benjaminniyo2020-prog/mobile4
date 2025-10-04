import os
import pandas as pd
import streamlit as st

# Path to model file (same folder as app.py)
model_file = os.path.join(os.path.dirname(__file__), "phone_sales_data.sav")

# Load the trained model
if not os.path.exists(model_file):
    st.error(f"❌ Model file '{model_file}' not found. Please upload it to the repo.")
    loaded_model = None
else:
    try:
        loaded_model = joblib.load(model_file)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        loaded_model = None


# Function to predict phone price
def phone_price_prediction(screen_size, ram, storage, battery_capacity, camera_quality):
    new_phone = pd.DataFrame([{
        'Screen Size (inches)': screen_size,
        'RAM (GB)': ram,
        'Storage (GB)': storage,
        'Battery Capacity (mAh)': battery_capacity,
        'Camera Quality (MP)': camera_quality
    }])
    predicted_price = loaded_model.predict(new_phone)
    return predicted_price[0]


# Main Streamlit app
def main():
    st.set_page_config(page_title="Phone Price Predictor", page_icon="📱", layout="centered")
    st.title("📱 Phone Price Prediction App")
    st.write("Enter phone specifications to predict its price.")

    # Input fields
    screen_size = st.text_input('Screen Size (inches)', '6.2')
    ram = st.text_input('RAM (GB)', '4')
    storage = st.text_input('Storage (GB)', '64')
    battery_capacity = st.text_input('Battery Capacity (mAh)', '4000')
    camera_quality = st.text_input('Camera Quality (MP)', '48')

    if st.button('🔮 Predict Price'):
        if loaded_model is None:
            st.error("⚠️ Model not loaded. Please upload the model file.")
            return

        try:
            # Convert inputs
            screen_size = float(screen_size)
            ram = int(ram)
            storage = int(storage)
            battery_capacity = int(battery_capacity)
            camera_quality = int(camera_quality)

            # Prediction
            price = phone_price_prediction(screen_size, ram, storage, battery_capacity, camera_quality)
            st.success(f"💰 Predicted Phone Price: **${price:.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}. Please enter valid numeric values.")


if __name__ == "__main__":
    main()
