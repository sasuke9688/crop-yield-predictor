Project Title: All-India Crop Yield Predictor

Summary: This project is a machine learning-powered web application designed to provide realistic crop yield forecasts for farmers and agricultural planners across India. Users can select a State, District, Season, and Crop to receive an instant prediction for yield (tonnes/hectare), total estimated production, and typical water requirements.

The core of the project is a specialized HistGradientBoostingRegressor model trained on a cleaned dataset of over 300,000 records. By strategically filtering out extreme-yield outliers (like Sugarcane), the model achieves a highly accurate Mean Absolute Error (MAE) of just 0.68 tonnes/hectare for common grains, pulses, and oilseeds.

The user interface is built with Streamlit, providing a simple, interactive, and responsive experience for all users.

Key Features:

Highly Accurate Predictions: Specialized model provides realistic forecasts for common crops with a very low average error (MAE ~0.68).

Dynamic UI: The "District" dropdown automatically updates based on the selected "State."

Actionable Insights: Provides three key metrics:

Predicted Yield (tonnes per hectare)

Estimated Production (total tonnes for the area)

Water Requirement (in million liters)

Data-Driven: Trained on the 'India Agriculture Crop Production' dataset, filtering over 320,000 valid records.

Technology Used:

Backend & Modeling: Python, scikit-learn, Pandas, NumPy

Web Framework: Streamlit

Deployment: Streamlit Community Cloud & GitHub
