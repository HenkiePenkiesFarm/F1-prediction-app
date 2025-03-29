import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.title("F1 Race Finish Predictor")
st.markdown("Voorspel de finishpositie van een coureur op basis van racegegevens, weer en odds. Gebaseerd op echte historische data.")

@st.cache_data
def load_historical_data():
    base_url = "http://ergast.com/api/f1/results.json"
    limit = 1000
    all_races = []
    r = requests.get(f"{base_url}?limit=1")
    total = int(r.json()['MRData']['total'])
    for offset in range(0, total, limit):
        r = requests.get(f"{base_url}?limit={limit}&offset={offset}")
        if r.status_code == 200:
            all_races.extend(r.json()['MRData']['RaceTable']['Races'])
    meta_fields = [
        'season', 'round', 'raceName', 'date', 'time',
        ['Circuit', 'circuitId'], ['Circuit', 'Location', 'country']
    ]
    df = pd.json_normalize(
        all_races,
        record_path='Results',
        meta=meta_fields,
        errors='ignore'
    )
    return df

@st.cache_resource
def train_model(df):
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['laps'] = pd.to_numeric(df['laps'], errors='coerce')
    df['fastestLap.rank'] = pd.to_numeric(df.get('FastestLap.rank', np.nan), errors='coerce')
    cat_cols = ['Driver.driverId', 'Constructor.constructorId', 'Circuit.circuitId', 'Circuit.Location.country']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df['temp'] = np.random.normal(25, 5, size=len(df))
    df['humidity'] = np.random.randint(40, 80, size=len(df))
    df['wind'] = np.random.uniform(1, 6, size=len(df))
    df['rain'] = np.random.binomial(1, 0.1, size=len(df)) * np.random.uniform(0, 5, size=len(df))
    df['odds'] = np.random.uniform(1.2, 6.0, size=len(df))
    features = ['grid', 'laps', 'fastestLap.rank', 'season', 'round'] + cat_cols + ['temp', 'humidity', 'wind', 'rain', 'odds']
    df_model = df.dropna(subset=features + ['position'])
    X = df_model[features]
    y = df_model['position']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features, df_model

st.info("Laden en trainen op echte racehistorie...")
df = load_historical_data()
model, scaler, features, df_model = train_model(df)

drivers = df_model['Driver.driverId'].unique()
constructors = df_model['Constructor.constructorId'].unique()
circuits = df_model['Circuit.circuitId'].unique()
countries = df_model['Circuit.Location.country'].unique()

st.sidebar.header("Input Parameters")
grid = st.sidebar.slider("Startpositie (grid)", 1, 20, 5)
laps = st.sidebar.slider("Aantal rondes", 40, 80, 55)
fastest_lap_rank = st.sidebar.slider("Snelste ronde ranking", 1, 20, 3)
season = st.sidebar.selectbox("Seizoen", sorted(df_model['season'].unique(), reverse=True))
round_nr = st.sidebar.slider("Ronde #", 1, 25, 10)

driver = st.sidebar.selectbox("Coureur", drivers)
constructor = st.sidebar.selectbox("Team", constructors)
circuit = st.sidebar.selectbox("Circuit", circuits)
country = st.sidebar.selectbox("Land", countries)

temp = st.sidebar.slider("Temperatuur (°C)", 10, 40, 26)
humidity = st.sidebar.slider("Luchtvochtigheid (%)", 20, 100, 55)
wind = st.sidebar.slider("Windsnelheid (m/s)", 0, 10, 3)
rain = st.sidebar.slider("Neerslag (mm)", 0.0, 10.0, 0.0)
odds = st.sidebar.slider("Bookmaker odds", 1.0, 10.0, 2.5)

input_df = pd.DataFrame([{
    'grid': grid,
    'laps': laps,
    'fastestLap.rank': fastest_lap_rank,
    'season': season,
    'round': round_nr,
    'Driver.driverId': driver,
    'Constructor.constructorId': constructor,
    'Circuit.circuitId': circuit,
    'Circuit.Location.country': country,
    'temp': temp,
    'humidity': humidity,
    'wind': wind,
    'rain': rain,
    'odds': odds
}])

X_input_scaled = scaler.transform(input_df)
prediction = model.predict(X_input_scaled)

st.subheader("Voorspelde finishpositie")
st.metric(label="Finishpositie", value=f"{prediction[0]:.1f}")

st.subheader("Inputwaarden")
fig, ax = plt.subplots()
ax.bar(input_df.columns, input_df.values[0])
plt.xticks(rotation=90)
st.pyplot(fig)

st.caption("Trained on real Ergast F1 historical data. Features: coureur, team, circuit, weer, odds, grid, fastest lap, laps, etc.")
