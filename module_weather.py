import pandas as pd
import numpy as np
import requests
import os

def fetch_and_save_weather(lat: float, lon: float, start_date: str, end_date: str, output_pkl: str, output_csv: str) -> pd.DataFrame:
    """
    Pobiera dane pogodowe z Open-Meteo API dla zadanego przedziału czasowego
    i zapisuje wyniki surowe (Raw Weather) do plików CSV oraz PKL w katalogu `data/`.
    """
    print("--- [MODULE METEO] Uruchamianie pobierania danych pogodowych ---")
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"],
        "timezone": "Europe/Warsaw"
    }

    print(f"Pobieranie faktycznych danych dla współrzędnych: ({lat}, {lon})...")
    response = requests.get(archive_url, params=params)
    response.raise_for_status()
    actual_data = response.json()

    df = pd.DataFrame({
        "date": pd.to_datetime(actual_data["daily"]["time"]),
        "actual_temp": actual_data["daily"]["temperature_2m_max"],
        "actual_precip": actual_data["daily"]["precipitation_sum"],
        "actual_wind": actual_data["daily"]["windspeed_10m_max"]
    })

    # Symulacja historycznych prognoz na podstawie błędu empirycznego
    print("Generowanie symulowanych prognoz meteorologicznych dla T-3 oraz T-14...")
    np.random.seed(42)
    
    # T-3: Prognoza na 3 dni przed przyjazdem - relatywnie dokładna
    df["forecast_3d_temp"] = df["actual_temp"] + np.random.normal(0, 1.5, len(df))
    df["forecast_3d_precip"] = np.maximum(0, df["actual_precip"] + np.random.normal(0, 2.0, len(df)))
    
    # T-14: Prognoza na 14 dni przed - obarczona znacznie większym błędem
    df["forecast_14d_temp"] = df["actual_temp"] + np.random.normal(0, 3.5, len(df))
    df["forecast_14d_precip"] = np.maximum(0, df["actual_precip"] + np.random.normal(0, 5.0, len(df)))
    
    # Oczyszczanie i zaokrąglanie wartości
    for col in ["forecast_3d_precip", "forecast_14d_precip"]:
        df[col] = df[col].round(1)
    for col in ["forecast_3d_temp", "forecast_14d_temp"]:
        df[col] = df[col].round(1)
        
    # Zapis danych do katalogu wyników
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df.to_pickle(output_pkl)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Sukces: Zapisano zestaw pogodowy do plików '{output_pkl}' oraz '{output_csv}'.")
    return df

if __name__ == "__main__":
    fetch_and_save_weather(53.80, 21.57, "2024-01-01", "2025-12-31", "data/raw_weather.pkl", "data/raw_weather.csv")
