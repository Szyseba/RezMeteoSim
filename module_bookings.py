import pandas as pd
import numpy as np
import os

def generate_and_save_bookings(weather_df: pd.DataFrame, n_bookings: int, output_pkl: str, output_csv: str) -> pd.DataFrame:
    """
    Generuje zbiór rezerwacji przypisując im odpowiednie parametry dystrybucyjne.
    Wymaga dostarczenia surowej ramki danych z modułu pogodowego w celu wylosowania dat.
    """
    print("--- [MODULE BOOKINGS] Uruchamianie generatora ruchu biznesowego ---")
    dates = weather_df['date']
    weights = np.ones(len(dates))
    
    # 1. Zwiększenie popytu na wakacje (Czerwiec - Wrzesień: 6-9)
    print("Obliczanie dystrybuant sezonowości i weekendów...")
    weights[dates.dt.month.isin([6, 7, 8, 9])] *= 3.0
    
    # 2. Zwiększenie popytu na weekendy (Piątek, Sobota, Niedziela: 4, 5, 6)
    weights[dates.dt.dayofweek.isin([4, 5, 6])] *= 1.8 
    
    weights /= weights.sum()
    arrivals = np.random.choice(dates, size=n_bookings, p=weights)
    
    # Rozkład log-normal dla zachowania asymetrii w prawo (lead time)
    lead_time_variance = np.random.lognormal(mean=2.5, sigma=0.9, size=n_bookings)
    lead_times = 14 + np.round(lead_time_variance).astype(int)
    
    # Rzut monetą elastycznej taryfy
    print(f"Generowanie rozkładu taryf dla N={n_bookings} wpisów...")
    is_flexible = np.random.choice([True, False], size=n_bookings, p=[0.7, 0.3])
    
    bookings_df = pd.DataFrame({
        'booking_id': range(1, n_bookings + 1),
        'arrival_date': arrivals,
        'lead_time': lead_times,
        'is_flexible': is_flexible
    })
    
    # Zapis danych do katalogu wyników
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    bookings_df.to_pickle(output_pkl)
    bookings_df.to_csv(output_csv, index=False)
    
    print(f"✅ Sukces: Zapisano zestaw rezerwacyjny do plików '{output_pkl}' oraz '{output_csv}'.")
    return bookings_df

if __name__ == "__main__":
    import sys
    # Symulacja na wypadek bezpośredniego wywołania pojedynczego pliku
    if os.path.exists("data/raw_weather.pkl"):
        weather = pd.read_pickle("data/raw_weather.pkl")
        generate_and_save_bookings(weather, 5000, "data/raw_bookings.pkl", "data/raw_bookings.csv")
    else:
        print("[OSTRZEŻENIE] Brak danych z Open-Meteo do wylosowania dat. Uruchom najpierw module_weather.py")
        sys.exit(1)
