import pandas as pd
import numpy as np
import os

def calculate_cancellations(bookings_df: pd.DataFrame, weather_df: pd.DataFrame, output_pkl: str, output_csv: str) -> pd.DataFrame:
    """
    Rdzeń ML: Łączy dwie ramki danych, wylicza logikę Szoku Pogodowego i ustala docelową
    odmowę przyjazdu (Churn Prediction) i zapisuje wynik do katalogu `data/`.
    """
    print("--- [MODULE ML/MERGE] Uruchamianie silnika decyzyjnego wektorów ---")
    
    # 1. Złączenie danych (LEFT JOIN na datach) (przyjazd + historyczna pogoda i prognozy t-3/t-14)
    print("Łączenie danych rezerwacyjnych z cechami (features) meteo...")
    df = bookings_df.merge(weather_df, left_on='arrival_date', right_on='date', how='left')
    
    # 2. Obliczanie kar z prognoz
    print("Wyliczanie wskaźników Penalty Score dla T-14 i T-3...")
    w1, w2 = 2.0, 0.5
    p_14d = w1 * df['forecast_14d_precip'] + w2 * np.maximum(0, 22 - df['forecast_14d_temp'])
    p_3d  = w1 * df['forecast_3d_precip']  + w2 * np.maximum(0, 22 - df['forecast_3d_temp'])
    
    df['weather_penalty_14d'] = p_14d.round(2)
    df['weather_penalty_3d']  = p_3d.round(2)
    
    # 3. Kalkulacja Szoku 
    df['weather_shock'] = np.maximum(0, p_3d - p_14d).round(2)
    
    # 4. Modelowanie Prawdopodobieństwa Sigmoidą Logistyczną
    print("Modelowanie i wyprowadzanie predykcji churn za pomocą błędu statystycznego...")
    base_prob = 0.05
    max_penalty_prob = 0.55  
    shock_threshold = 4.0    
    steepness = 1.2          
    
    # Prawdopodobieństwo skacze w górę gdy szok zrówna się i przekroczy próg
    logistic_increase = 1 / (1 + np.exp(-steepness * (df['weather_shock'] - shock_threshold)))
    
    cancel_prob = np.where(
        df['is_flexible'], 
        base_prob + max_penalty_prob * logistic_increase,
        base_prob
    )
    df['cancellation_prob'] = cancel_prob
    
    # 5. Binaryzacja anulacji 
    df['is_canceled'] = (np.random.rand(len(df)) < cancel_prob).astype(int)
    
    # 6. Czyszczenie stempla danych dla ładniejszego podglądu
    cols_order = [
        'booking_id', 'arrival_date', 'lead_time', 'is_flexible', 
        'actual_temp', 'actual_precip', 'forecast_14d_temp', 'forecast_14d_precip', 
        'forecast_3d_temp', 'forecast_3d_precip', 'weather_penalty_14d', 
        'weather_penalty_3d', 'weather_shock', 'cancellation_prob', 'is_canceled'
    ]
    df = df[cols_order]
    
    # Zapis danych
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df.to_pickle(output_pkl)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Sukces: Zapisano scalony Master-dataset do plików '{output_pkl}' oraz '{output_csv}'.")
    return df

if __name__ == "__main__":
    import sys
    if os.path.exists("data/raw_weather.pkl") and os.path.exists("data/raw_bookings.pkl"):
        weather = pd.read_pickle("data/raw_weather.pkl")
        bookings = pd.read_pickle("data/raw_bookings.pkl")
        calculate_cancellations(bookings, weather, "data/final_hotel_dataset.pkl", "data/final_hotel_dataset.csv")
    else:
        print("[BŁĄD] Uruchom wcześniejsze moduły w celu zapisu stanu do folderu data/ !")
        sys.exit(1)
