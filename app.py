import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import zipfile  # <-- Make sure this import is at the top
import io       # <-- Make sure this import is at the top

# --- 1. SET UP PAGE CONFIGURATION ---
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

# --- 2. DEFINE FILE PATHS ---
MODEL_PATH = "crop_yield_model_specialized_v2.pkl"
ENCODERS_PATH = "encoders_specialized_v2.pkl"

# --- THIS IS THE FIX ---
# 1. The exact name of the zip file you uploaded (from your screenshot)
DATA_ZIP_PATH = "India Agriculture Crop Production dataset.zip" 
# 2. The exact name of the CSV file *inside* that zip file (from your screenshot)
DATA_CSV_NAME = "India Agriculture Crop Production dataset/India Agriculture Crop Production.csv"
# ------------------------------------

# Use st.cache_resource to load these only once
@st.cache_resource
def load_artifacts():
    """
    Loads the saved model, encoders, and raw data from a ZIP FILE.
    """
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        return None, None, None, None
    
    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: Encoders file not found at {ENCODERS_PATH}")
        return None, None, None, None
        
    try:
        # --- Read from the zip file ---
        with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as z:
            # Open the CSV file from within the zip
            with z.open(DATA_CSV_NAME) as f:
                # Read it into pandas
                df_raw = pd.read_csv(io.TextIOWrapper(f, 'utf-8'))
                df_raw = df_raw.rename(columns=lambda x: x.strip())
    except FileNotFoundError:
        st.error(f"Error: Data ZIP file not found at {DATA_ZIP_PATH}")
        return None, None, None, None
    except KeyError:
        st.error(f"Error: Could not find file '{DATA_CSV_NAME}' inside the zip.")
        return None, None, None, None
    
    # Extract encoders
    enc_state = encoders["state"]
    enc_district = encoders["district"]
    enc_crop = encoders["crop"]
    enc_season = encoders["season"]
    
    # Get lists for dropdowns
    states = sorted(list(enc_state.classes_))
    seasons = sorted(list(enc_season.classes_))
    crops = sorted(list(enc_crop.classes_))
    
    # Get the yield cutoff and outlier crops info from the model/data
    yield_cutoff = 25.67 
    outlier_crops = ['Coconut', 'Sugarcane', 'Banana', 'Onion', 'Sweet potato', 
                     'Potato', 'Tapioca', 'Sunflower', 'Sannhamp', 'Arhar/Tur', 
                     'Ginger', 'Other Rabi pulses', 'Cashewnut', 'Maize', 'Groundnut', 
                     'Guar seed', 'Sesamum', 'other oilseeds', 'Niger seed', 'Turmeric', 
                     'Tobacco', 'Dry chillies', 'Jute', 'Peas & beans (Pulses)', 
                     'Cotton(lint)', 'Mesta', 'Rice'] 

    return model, enc_state, enc_district, enc_crop, enc_season, df_raw, states, seasons, crops, yield_cutoff, outlier_crops

# Load everything
(model, enc_state, enc_district, enc_crop, enc_season, 
 df_raw, states, seasons, crops, 
 yield_cutoff, outlier_crops) = load_artifacts()

# --- 3. DEFINE WATER REQUIREMENTS ---
crop_water = {
    "Rice": 1200, "Wheat": 450, "Sugarcane": 1500, "Maize": 600,
    "Cotton": 700, "Groundnut": 550, "Pulses": 400, "Potato": 500,
    "Soybean": 500, "Barley": 450, "Arecanut": 1000, "Coconut": 1100,
    "Tea": 1200, "Coffee": 800, "Sunflower": 600, "Banana": 1600, "Other": 700
}

# --- 4. BUILD THE USER INTERFACE (UI) ---

st.title("🌾 India Crop Yield Predictor")
st.markdown("Select your location, crop, and area to get a yield prediction.")

if model is not None: # Only run if models loaded successfully
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # --- State Selection ---
        selected_state = st.selectbox("1. Select a State:", states)
        
        # --- District Selection (Dynamic) ---
        if selected_state:
            # Get districts for the selected state that are *also* in the encoder
            known_districts = set(enc_district.classes_)
            available_districts = sorted(
                list(df_raw[df_raw["State"] == selected_state]["District"].dropna().unique())
            )
            # Filter to only districts the model was trained on
            districts_for_state = [d for d in available_districts if d in known_districts]
            
            if districts_for_state:
                selected_district = st.selectbox("2. Select a District:", districts_for_state)
            else:
                st.error(f"No valid districts found for {selected_state} in the model's training data.")
                selected_district = None
        else:
            selected_district = None

    with col2:
        # --- Season Selection ---
        selected_season = st.selectbox("3. Select a Season:", seasons)
        
        # --- Crop Selection ---
        selected_crop = st.selectbox("4. Select a Crop:", crops)
        
    # --- Area Input ---
    selected_area = st.number_input("5. Enter Area (in hectares):", min_value=0.01, value=1.0, step=0.1)
    
    st.markdown("---")

    # --- Prediction Button ---
    if st.button("✨ Predict Yield", type="primary"):
        if selected_district and selected_season and selected_crop and selected_area:
            
            # --- 5. ENCODE INPUTS AND PREDICT ---
            try:
                # Transform text inputs to encoded numbers
                s_encoded = enc_state.transform([selected_state])[0]
                d_encoded = enc_district.transform([selected_district])[0]
                c_encoded = enc_crop.transform([selected_crop])[0]
                sea_encoded = enc_season.transform([selected_season])[0]

                # Prepare features for model (MUST match training columns)
                features = ["State_encoded", "District_encoded", "Season_encoded", "Crop_encoded", "Area"]
                features_df = pd.DataFrame(
                    [[s_encoded, d_encoded, sea_encoded, c_encoded, selected_area]],
                    columns=features
                )

                # Predict yield
                pred_yield = model.predict(features_df)[0]
                pred_yield = max(0, pred_yield) # Ensure yield is not negative
                pred_prod = pred_yield * selected_area
                
                # Get water req
                water_mm = crop_water.get(selected_crop, crop_water["Other"])
                water_liters = water_mm * 10000 * selected_area

                # --- 6. DISPLAY RESULTS ---
                st.subheader("🎉 Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("🌾 Predicted Yield", f"{pred_yield:.2f} t/ha")
                    st.metric("💧 Water Required", f"{water_liters/1e6:.2f} M Liters")
                with res_col2:
                    st.metric("📦 Estimated Production", f"{pred_prod:.2f} tonnes")
                
                st.info(f"**Location:** {selected_district}, {selected_state}\n\n"
                        f"**Crop:** {selected_crop} ({selected_season})")

                # Show the warning about model specialization
                st.warning(f"""
                **Model Note:** This model is specialized for common crops and grains (yields < {yield_cutoff:.2f} t/ha). 
                Its predictions for very high-yield crops (like Sugarcane, Banana, etc.) may be less accurate.
                """)

            except ValueError as e:
                st.error(f"Error during encoding or prediction: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.error("Please fill in all the fields.")
else:
    st.error("Model artifacts could not be loaded. The app cannot run.")
