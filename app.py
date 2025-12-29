import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
import joblib  # For loading my saved RF model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from streamlit.components.v1 import html as st_html  # Fixed import


# Load the trained Random Forest model
@st.cache_resource
def load_model():
    return joblib.load('/Users/deen/Desktop/crime analysis/rf_model.pkl')
    
rf_model = load_model()

# App title
st.title("Coventry Crime Hotspot Analysis Dashboard")
st.markdown("Interactive exploration of crime patterns, clustering, and predictive modelling")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a view", 
    ["Total Crime Map", "K-Means Clustering", "DBSCAN Clustering", "Random Forest Prediction Map", "Predict Crime Risk"])

# Function to embed HTML map
def embed_map(html_file, height=700):
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            map_html = f.read()
        st_html(map_html, height=height, scrolling=True)
    except FileNotFoundError:
        st.error(f"Map file '{html_file}' not found. Place it in the same folder as app.py")
        
if page == "Total Crime Map":
    st.header("Total Recorded Crimes by LSOA (2018–2025)")
    embed_map("/Users/Deen/Desktop/crime analysis/map/coventry_total_crime_interactive_map.html")

elif page == "K-Means Clustering":
    st.header("K-Means Clustering of Crime & Deprivation (2019)")
    embed_map("/Users/Deen/Desktop/crime analysis/map/coventry_kmeans_clusters_2019.html")

elif page == "DBSCAN Clustering":
    st.header("DBSCAN Density Clustering & Outliers (2019)")
    st.write("Outliers (light cyan) often near nightlife venues – see Appendix for OSM examples.")
    embed_map("/Users/Deen/Desktop/crime analysis/map/DBSCAN_clusters_2019.html")
    
        
elif page == "Random Forest Prediction Map":
    st.header("Random Forest Predicted High-Risk Hotspots (2019 Violence & Sexual Offences)")
    embed_map("/Users/Deen/Desktop/crime analysis/map/coventry_predicted_hotspots_2019.html")

elif page == "Predict Crime Risk":
    st.header("Predict if an LSOA is High-Risk (Random Forest)")
    st.write("Enter 2019 IMD deprivation scores (0–1 scale, higher = more deprived)")

    # Adjust these feature names to exactly match your training data
    income = st.slider("Income Score (rate)", 0.0, 1.0, 0.3)
    education = st.slider("Education, Skills & Training Score", 0.0, 1.0, 0.4)
    health = st.slider("Health Deprivation & Disability Score", 0.0, 1.0, 0.5)
    housing = st.slider("Barriers to Housing & Services Score", 0.0, 1.0, 0.3)
    living_env = st.slider("Population aged 16-59: mid 2015 (excluding prisoners)", 0.0, 1.0, 0.2)
    children = st.slider("Children and Young People Sub-domain Score", 0.0, 1.0, 0.4)

    if st.button("Predict Risk"):
        # Create input DataFrame – column order MUST match training
        features = pd.DataFrame([[income, education, health, housing, living_env, children]],
                                columns=["Income Score (rate)", "Education, Skills and Training Score",
                                            "Health Deprivation and Disability Score",
                                            "Barriers to Housing and Services Score",
                                            "Children and Young People Sub-domain Score",
                                            "Population aged 16-59: mid 2015 (excluding prisoners)"])
        
        prediction = rf_model.predict(features)[0]
        probability = rf_model.predict_proba(features)[0][1] * 100

        if prediction == 1:
            st.error(f"**HIGH CRIME RISK** (Probability: {probability:.1f}%)")
        else:
            st.success(f"**Low Crime Risk** (Probability: {100 - probability:.1f}%)")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("implementation  – Coventry Crime Analysis 2025")