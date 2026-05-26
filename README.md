# 🏙️ Crime in a Cultural City
### A Spatial-Temporal Analysis of Coventry (2018–2025)

> Exploring how the UK City of Culture 2021 designation — and a global pandemic — reshaped crime patterns across one of England's most historically rich cities.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://coventry-crime-analysis-prediction.streamlit.app/)
[![Interactive Maps](https://img.shields.io/badge/Interactive_Maps-GitHub_Pages-222?style=flat-square&logo=github)](https://findmeanickpls.github.io/Coventry-crime-analysis/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What's This About?

Coventry was named UK City of Culture in 2017, launched its cultural programme in May 2021 (delayed by COVID-19), and wrapped up in May 2022. That overlap — a city trying to celebrate itself during a pandemic recovery — made for a genuinely unusual set of conditions to study.

This project uses **seven years of open police data**, **socioeconomic deprivation indices**, and a mix of GIS, machine learning, and clustering techniques to ask: *did any of this actually affect crime?*

Short answer: yes, in complicated ways. Crime didn't simply go up or down — different types moved in opposite directions depending on the period, the neighbourhood, and what was happening in the city at the time.

---

## Live Demos

| Tool | Link |
|------|------|
| 🗺️ Interactive Folium Maps | [findmeanickpls.github.io/Coventry-crime-analysis](https://findmeanickpls.github.io/Coventry-crime-analysis/) |
| 📊 Streamlit Prediction Dashboard | [coventry-crime-analysis-prediction.streamlit.app](https://coventry-crime-analysis-prediction.streamlit.app/) |

The Streamlit app lets you input deprivation scores and get a Random Forest–based high/low crime risk prediction for any LSOA profile.

---

## Key Findings

**Temporal trends (2018–2025)**
- Crime in Coventry peaked at over **42,000 incidents in 2022** — coinciding with full pandemic reopening and the City of Culture programme
- During lockdowns, property crimes (burglary −33%, bicycle theft −29%, shoplifting −25%) dropped sharply, while public order offences surged +93% and drug-related crime rose +68%
- Monthly counts stayed consistently above 3,500 throughout the City of Culture year; by early 2025 they had fallen back towards ~2,600–2,800

**Spatial patterns**
- High-crime LSOAs clustered persistently in central and northern Coventry: **Hillfields, St. Michael's, Foleshill, Wood End**
- Strong positive correlations between crime and health deprivation (+0.32), housing barriers (+0.31), and education scores (+0.25) at neighbourhood level

**Machine learning**
- Random Forest classified high-crime LSOAs with **81% accuracy** (F1: 0.80), using 2019 IMD deprivation indicators as features
- Top predictors: income score, health deprivation, education and skills
- DBSCAN identified 99 spatial outliers — many near nightlife venues — suggesting crime generators that operate independently of structural deprivation

---

## Project Structure

```
coventry-crime-analysis/
│
├── data/
│   ├── raw/                    # UK Police API downloads (2018–2025)
│   ├── processed/              # Cleaned & harmonised datasets
│   └── shapefiles/             # LSOA 2021 boundaries + lookup files
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_temporal_analysis.ipynb
│   ├── 03_spatial_analysis.ipynb
│   ├── 04_machine_learning.ipynb
│   └── 05_clustering.ipynb
│
├── maps/                       # Exported Folium HTML maps
│   ├── total_crimes_choropleth.html
│   ├── kmeans_clusters.html
│   ├── dbscan_clusters.html
│   └── random_forest_predictions.html
│
├── streamlit_app/
│   └── app.py                  # Streamlit prediction dashboard
│
├── figures/                    # Static charts and visualisations
└── README.md
```

---

## Stack

| Category | Tools |
|----------|-------|
| Data wrangling | `pandas`, `numpy` |
| Geospatial | `geopandas`, `shapely`, `folium` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` (Random Forest, Logistic Regression, K-Means, DBSCAN) |
| Web app | `streamlit` |
| Data sources | [UK Police API](https://data.police.uk/), [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) |

---

## Data Sources

- **Crime data**: UK Police open data API — ~280,000 street-level crime records for Coventry, January 2018 to May 2025
- **Deprivation**: [Index of Multiple Deprivation (IMD) 2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) — income, employment, education, health, housing and crime sub-domains at LSOA level
- **Boundaries**: ONS 2021 LSOA shapefiles with 2011–2021 lookup for boundary harmonisation

All data is open-source and anonymised. No individual-level records are used.

---

## Running Locally

**Clone and install**
```bash
git clone https://github.com/findmeanickpls/Coventry-crime-analysis.git
cd Coventry-crime-analysis
pip install -r requirements.txt
```

**Run the notebooks** (in order, or jump to whichever analysis interests you)
```bash
jupyter notebook notebooks/
```

**Launch the Streamlit app**
```bash
cd streamlit_app
streamlit run app.py
```

---

## Methodology at a Glance

```
UK Police API + ONS IMD
        ↓
   Data Cleaning
   (deduplication, LSOA boundary harmonisation 2011→2021, date parsing)
        ↓
   Feature Engineering
   (crime rate per 1,000 pop, seasonal labels, high/low binary target)
        ↓
   Exploratory Analysis
   (temporal trends, choropleth maps, correlation heatmaps)
        ↓
   Supervised ML                    Unsupervised Clustering
   Logistic Regression (76%)        K-Means (k=3)
   Random Forest (81%)              DBSCAN (outlier detection)
        ↓
   Predictive Maps + Streamlit Dashboard
```

Feature importance ranking (Random Forest, 2019 violence & sexual offences):

1. Income Score
2. Health Deprivation & Disability
3. Education, Skills & Training
4. Children & Young People Sub-domain
5. Barriers to Housing & Services
6. Population (aged 16–59)

---

## Limitations Worth Knowing

- Police-recorded data only captures *reported* crime — some offence types and communities are likely underrepresented
- IMD data is from 2019 and may not fully reflect how deprivation shifted during and after the pandemic
- Models were trained on 2019 violence & sexual offences specifically; generalisation to other years or crime types would need retraining
- Minor LSOA boundary mismatches from the 2011→2021 harmonisation were handled as nulls

---

## If You Want to Extend This

Some natural next steps if you're picking this up:

- Retrain models on 2022–2024 data to compare post-culture period predictions
- Add time-series forecasting (e.g. Prophet or ARIMA) for month-level crime volume
- Incorporate real-time police data feed updates into the Streamlit app
- Apply transfer learning to test if the model holds for similar mid-sized UK cities (e.g. Wolverhampton, Derby)
- Use updated IMD 2023 data when available

---

## Acknowledgements

Data from the [UK Police Open Data Portal](https://data.police.uk/) and the [Office for National Statistics](https://www.ons.gov.uk/). Thanks to everyone who keeps open public data actually open.

---

*Built as part of an MSc dissertation at the University of Bradford, 2025. Supervisor: Dr Pritesh Mistry.*
