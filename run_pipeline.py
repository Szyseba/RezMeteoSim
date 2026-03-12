import os
import time

# Import podkomponentów (modułów) ETL
from module_weather import fetch_and_save_weather
from module_bookings import generate_and_save_bookings
from module_merge_cancellations import calculate_cancellations

def main():
    """
    Główny orkiestrator (Pipeline) danych.
    Organizuje folder zrzutu `data/` i uruchamia kolejne węzły w logicznym rygorze ETL.
    """
    # 1. Definicja Katalogu Roboczego i Plików
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    RAW_WEATHER_PKL = os.path.join(DATA_DIR, "raw_weather.pkl")
    RAW_WEATHER_CSV = os.path.join(DATA_DIR, "raw_weather.csv")
    
    RAW_BOOKING_PKL = os.path.join(DATA_DIR, "raw_bookings.pkl")
    RAW_BOOKING_CSV = os.path.join(DATA_DIR, "raw_bookings.csv")
    
    FINAL_PKL = os.path.join(DATA_DIR, "final_hotel_dataset.pkl")
    FINAL_CSV = os.path.join(DATA_DIR, "final_hotel_dataset.csv")

    # Ustawienia parametrów modelu 
    LAT, LON = 53.80, 21.57 # Mikołajki
    START_DATE, END_DATE = "2020-01-01", "2025-12-31"
    N_BOOKINGS = 10000 # Liczba rezerwacji 

    print("======================================================")
    print("   ROZPOCZĘCIE PRZETWARZANIA POTOKU DANYCH (ETL)")
    print("======================================================")
    start_time = time.time()

    # Krok [1/3]: Pobieranie pogody
    print("\n>>> KROK 1: EKSTRAKCJA POGODY (EXTRACT)")
    weather_df = fetch_and_save_weather(
        LAT, LON, START_DATE, END_DATE, 
        RAW_WEATHER_PKL, RAW_WEATHER_CSV
    )

    # Krok [2/3]: Generowanie Ruchu (Rezerwacji)
    print("\n>>> KROK 2: TRANSFORMACJA POPYTU (TRANSFORM - DEMAND)")
    bookings_df = generate_and_save_bookings(
        weather_df, N_BOOKINGS, 
        RAW_BOOKING_PKL, RAW_BOOKING_CSV
    )

    # Krok [3/3]: Złączanie Zbiorów (Merge) i Wynik (Churn Prediction)
    print("\n>>> KROK 3: ŁĄCZENIE ZBIORÓW I WNIOSKOWANIE O ANULACJACH (LOAD & ML)")
    output_df = calculate_cancellations(
        bookings_df, weather_df, 
        FINAL_PKL, FINAL_CSV
    )
    
    elapsed = time.time() - start_time
    print("\n======================================================")
    print(f"✅ PIPELINE ZAKOŃCZONY SUKCESEM w czasie {elapsed:.2f} sek.")
    print("======================================================")
    
    print(f"Wygenerowane pliki pośrednie i końcowe znajdziesz w folderze: `{DATA_DIR}/`:\n")
    for file in os.listdir(DATA_DIR):
        print(f" -> {file}")
        
    print("\n--- Podgląd Finalnego Zbioru (head 5) ---")
    print(output_df.head(5).to_string())

if __name__ == "__main__":
    main()
