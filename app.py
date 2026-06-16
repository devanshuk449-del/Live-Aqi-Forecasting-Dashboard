import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import datetime
from datetime import date

# ML & Deep Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Attention, Conv1D, GlobalAveragePooling1D, 
    MultiHeadAttention, LayerNormalization, Add, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


print("\n--- HARDWARE ACCELERATION CHECK ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ APPLE SILICON GPU UNLOCKED: {len(gpus)} GPU(s) detected.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("❌ WARNING: GPU NOT FOUND. Deep Learning will run on CPU.")
print("-----------------------------------\n")

st.set_page_config(page_title="Global Live AQI Forecaster", layout="wide")


# CORE DICTIONARIES & FUNCTIONS
# The 6 Target Cities and their exact GPS Coordinates
CITY_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Chicago": {"lat": 41.8781, "lon": -87.6298}
}


# CUSTOM PHYSICS LOSS FUNCTION (GRADIENT-SAFE)

@tf.function
def physics_informed_loss(y_true, y_pred):
    """
    Punishes impossible atmospheric jumps, but uses Huber-style 
    error bounding to prevent Exploding Gradients in deep networks.
    """
    # Use Huber logic instead of pure MSE to prevent explosions
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= 1.0
    safe_loss = tf.where(is_small_error, 0.5 * tf.square(error), tf.abs(error) - 0.5)
    base_loss = tf.reduce_mean(safe_loss)
    
    # Physics Penalty
    predicted_jumps = tf.abs(y_pred[1:] - y_pred[:-1])
    physics_penalty = tf.reduce_mean(tf.maximum(0.0, predicted_jumps - 0.2)) 
    
    return base_loss + (0.5 * physics_penalty)

@st.cache_data(show_spinner=False)
def fetch_historical_training_data(city_name):
    """Instantly pulls hourly weather & AQI data (Jan 1 2023 to Today)"""
    lat = CITY_COORDS[city_name]["lat"]
    lon = CITY_COORDS[city_name]["lon"]
    today = date.today().strftime('%Y-%m-%d')
    start_date = "2023-01-01"

    # 1. Fetch Weather Archive
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={today}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    w_data = requests.get(weather_url).json()
    
    # 2. Fetch AQI Archive
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={today}&hourly=us_aqi"
    a_data = requests.get(aqi_url).json()

    # Build DataFrame
    df = pd.DataFrame({
        "datetime": pd.to_datetime(w_data["hourly"]["time"]),
        "temperature": w_data["hourly"]["temperature_2m"],
        "humidity": w_data["hourly"]["relative_humidity_2m"],
        "wind_speed": w_data["hourly"]["wind_speed_10m"],
        "aqi": a_data["hourly"]["us_aqi"]
    })
    
    # Clean data (Forward/Backward fill any missing API gaps)
    df = df.ffill().bfill()
    return df

def fetch_live_waqi_data(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        response = requests.get(url, timeout=10)
        
        # 1. Check if the server actually responded successfully
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: {response.text[:100]}"
            
        # 2. Safely attempt to parse the JSON
        try:
            data = response.json()
        except ValueError:
            return None, f"Non-JSON response received. Raw output: {response.text[:100]}"

        # 3. Process the data if the status is 'ok'
        if data.get('status') == 'ok':
            iaqi = data['data']['iaqi']
            live_data = {
                'temperature': iaqi.get('t', {}).get('v', 0.0),
                'humidity': iaqi.get('h', {}).get('v', 0.0),
                'wind_speed': iaqi.get('w', {}).get('v', 0.0),
                'visibility': 10.0, 
                'aqi': float(data['data']['aqi'])
            }
            return live_data, data['data']['time']['s']
        else:
            return None, f"WAQI Internal Error: {data.get('data', 'Unknown error')}"
            
    except Exception as e:
        return None, f"Connection failed: {str(e)}"

def add_rolling_features(df, target_col, windows=[3, 6, 12]):
    for w in windows:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].rolling(w, min_periods=1).mean()
        df[f'{target_col}_roll_std_{w}']  = df[target_col].rolling(w, min_periods=1).std().fillna(0)
    return df

def add_time_features(df):
    dt = df['datetime']
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    return df

def create_sequences(features, target, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)


# FRONTEND UI

st.title("🌍 Global API-Driven AQI Forecaster")

st.sidebar.header("📡 Live Data Connection")
selected_city = st.sidebar.selectbox("Select Target City", list(CITY_COORDS.keys()))

try:
    waqi_token = st.secrets["WAQI_API_TOKEN"]
    st.sidebar.success("Backend API Token Authenticated ✅")
except KeyError:
    waqi_token = None
    st.sidebar.error("Backend API Token Missing! Check .streamlit/secrets.toml")

st.sidebar.markdown("---")
st.sidebar.header("🔬 Model Configuration")
model_choice = st.sidebar.selectbox("Select Predictive Architecture", [
    "Temporal Fusion Transformer (TFT-Lite)",
    "Physics-Informed CNN-Attention (Deep Learning)", 
    "XGBoost", 
    "Stacking Ensemble (Mega-Stack)", 
    "Random Forest"
])
seq_len = st.sidebar.slider("Sequence Length (Hours)", 12, 72, 24)
ablation_weather = st.sidebar.checkbox("Include Weather Features", value=True)


# MAIN EXECUTION BLOCK

if st.button(f"Fetch Data & Train {model_choice} for {selected_city}"):
    
    with st.spinner(f"Downloading historical data for {selected_city} from 01/01/2023 to Today..."):
        raw_df = fetch_historical_training_data(selected_city)
        st.success(f"✅ Downloaded {len(raw_df):,} hours of historical data for {selected_city}!")
        
        # Feature Engineering
        raw_df = add_rolling_features(raw_df, 'aqi')
        raw_df = add_time_features(raw_df)
        
        numeric_df = raw_df.select_dtypes(include=[np.number]).copy()
        target_col = 'aqi'
        weather_cols = ['temperature', 'humidity', 'wind_speed']
        
        if ablation_weather:
            feature_cols = [c for c in numeric_df.columns if c != target_col]
        else:
            feature_cols = [c for c in numeric_df.columns if c not in weather_cols and c != target_col]

    with st.spinner(f"Training AI on {selected_city}'s atmospheric physics..."):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaled_X = scaler_X.fit_transform(numeric_df[feature_cols].astype('float32'))
        scaled_y = scaler_y.fit_transform(numeric_df[[target_col]].astype('float32'))

        X, y = create_sequences(scaled_X, scaled_y, seq_len)
        split_idx = int(len(X) * 0.8)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat  = X_test.reshape(X_test.shape[0], -1)

        # --- MODEL TRAINING ---
        if model_choice == "Temporal Fusion Transformer (TFT-Lite)":
            d_model = 64      
            num_heads = 4     
            lstm_units = 128
            drop_rate = 0.3

            inputs = Input(shape=(seq_len, len(feature_cols)))
            x = Dense(d_model, activation='relu')(inputs)
            lstm_out = LSTM(lstm_units, return_sequences=True)(x)
            attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)(lstm_out, lstm_out)
            res_out = Add()([lstm_out, attention_out])
            res_out = LayerNormalization()(res_out)
            pooled = GlobalAveragePooling1D()(res_out)
            dense = Dense(64, activation='relu')(pooled)
            dense = Dropout(drop_rate)(dense)
            outputs = Dense(1)(dense)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(delta=1.0))
            callbacks = [EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]
            model.fit(X_train, y_train, epochs=35, batch_size=256, verbose=1, validation_split=0.15, callbacks=callbacks)

        elif model_choice == "Physics-Informed CNN-Attention (Deep Learning)":
            inputs = Input(shape=(seq_len, len(feature_cols)))
            
            # 1. Spatial/Weather Feature Extraction
            x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x) # <--- SPEED BUMP 1
            x = Dropout(0.2)(x)
            
            # 2. Temporal Sequence Learning
            lstm_out = LSTM(128, return_sequences=True)(x)
            
            # 3. Multi-Head Temporal Attention
            attention_out = MultiHeadAttention(num_heads=4, key_dim=128)(lstm_out, lstm_out)
            
            # 4. Pooling & Deep Dense Output
            pooled = GlobalAveragePooling1D()(attention_out)
            
            dense = Dense(128, activation='relu')(pooled)
            dense = BatchNormalization()(dense) # <--- SPEED BUMP 2
            dense = Dropout(0.3)(dense)
            
            dense = Dense(64, activation='relu')(dense)
            dense = BatchNormalization()(dense) # <--- SPEED BUMP 3
            
            # Sigmoid guarantees the AI's internal answer stays between 0.0 and 1.0
            # preventing the 1066 hallucination.
            outputs = Dense(1, activation='sigmoid')(dense) 
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=physics_informed_loss)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
            ]
            
            model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.15, callbacks=callbacks)

        elif model_choice == "XGBoost":
            model = XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
            model.fit(X_train_flat, y_train.ravel())

        elif model_choice == "Stacking Ensemble (Mega-Stack)":
            base_models = [('rf', RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1))]
            meta_learner = XGBRegressor(n_estimators=50, max_depth=4, n_jobs=-1)
            model = StackingRegressor(estimators=base_models, final_estimator=meta_learner, cv=2, n_jobs=-1)
            model.fit(X_train_flat, y_train.ravel())

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1)
            model.fit(X_train_flat, y_train.ravel())

    
    # LIVE INFERENCE
    
    st.markdown("---")
    if waqi_token:
        st.info(f"Connecting to WAQI Live API for {selected_city.upper()}...")
        # Note: WAQI uses lowercase slugs for cities
        live_data, timestamp = fetch_live_waqi_data(selected_city.lower(), waqi_token)
        
        if live_data:
            st.success(f"Live API Data Captured at {timestamp}")
            st.json(live_data)
            
            future_hours = 24
            base_sequence = numeric_df[feature_cols].iloc[-seq_len:].copy().astype('float64')
            
            # --- FIX 1: Smooth the Temporal Shock (Weather) ---
            if ablation_weather:
                for col in ['temperature', 'humidity', 'wind_speed']:
                    if col in base_sequence.columns and col in live_data:
                        live_val = live_data[col]
                        for j in range(1, 7): 
                            idx = -j
                            if abs(idx) <= len(base_sequence):
                                factor = (7 - j) / 6.0 
                                old_val = base_sequence.iloc[idx, base_sequence.columns.get_loc(col)]
                                base_sequence.iloc[idx, base_sequence.columns.get_loc(col)] = (live_val * factor) + (old_val * (1 - factor))
            
            # --- FIX 2: Smooth the "Ghost of Winter" (AQI Memory) ---
            live_aqi = live_data.get('aqi', 50.0)
            for col in base_sequence.columns:
                if 'aqi_roll' in col: 
                    for j in range(1, 7): 
                        idx = -j
                        if abs(idx) <= len(base_sequence):
                            factor = (7 - j) / 6.0
                            old_val = base_sequence.iloc[idx, base_sequence.columns.get_loc(col)]
                            base_sequence.iloc[idx, base_sequence.columns.get_loc(col)] = (live_aqi * factor) + (old_val * (1 - factor))
            
            future_preds = []
            current_time = pd.Timestamp.now()
            
            for i in range(future_hours):
                current_seq_scaled = scaler_X.transform(base_sequence.astype('float32'))
                current_seq_scaled = np.clip(current_seq_scaled, 0.0, 1.0) 
                
                if model_choice in ["Physics-Informed CNN-Attention (Deep Learning)", "Temporal Fusion Transformer (TFT-Lite)"]:
                    p_scaled = model.predict(current_seq_scaled.reshape(1, seq_len, len(feature_cols)), verbose=0)[0][0]
                else:
                    p_scaled = model.predict(current_seq_scaled.reshape(1, -1))[0]

                future_preds.append(p_scaled)
                
                next_step = base_sequence.iloc[-1].copy()
                next_time = current_time + pd.Timedelta(hours=i+1)
                
                # Step time forward dynamically
                if 'hour_sin' in next_step.index:
                    next_step['hour_sin'] = np.sin(2 * np.pi * next_time.hour / 24)
                    next_step['hour_cos'] = np.cos(2 * np.pi * next_time.hour / 24)
                    
                base_sequence = pd.concat([base_sequence.iloc[1:], pd.DataFrame([next_step])], ignore_index=True)

            future_actual = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).ravel()

            # --- BUILD THE PREDICTION TABLE & CHART ---
            future_times = [(current_time + pd.Timedelta(hours=i+1)).strftime('%I:00 %p') for i in range(future_hours)]
            forecast_df = pd.DataFrame({"Time": future_times, "Predicted AQI": np.round(future_actual, 1)})
            forecast_df.set_index("Time", inplace=True)

            st.markdown(f"#### Live 24-Hour Projection: {selected_city.upper()}")
            col_table, col_chart = st.columns([1, 3])
            
            with col_table:
                st.dataframe(forecast_df, height=400, width='stretch')
                
            with col_chart:
                fig_forecast = px.area(y=future_actual, markers=True)
                fig_forecast.update_layout(
                    xaxis_title="Hours from Now", 
                    yaxis_title="Predicted AQI Level", 
                    template="plotly_white", 
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_forecast, width='stretch')
        else:
            st.error(timestamp)