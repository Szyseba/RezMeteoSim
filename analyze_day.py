import pandas as pd
import argparse
import sys
from datetime import datetime

def interpret_day(input_file: str, target_date: str):
    """
    Wczytuje plik .pkl i interpretuje pojedynczą rezerwację z wybranego dnia
    w sposób przyjazny dla człowieka.
    """
    # 1. Wczytanie danych
    try:
        df = pd.read_pickle(input_file)
    except FileNotFoundError:
        print(f"\n[BŁĄD] Nie znaleziono pliku: {input_file}")
        print("Uruchom najpierw skrypt `generate_hotel_data.py` aby go utworzyć.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[BŁĄD] Wystąpił problem z wczytaniem pliku PKL: {e}")
        sys.exit(1)
        
    # 2. Walidacja formatu daty (RRRR-MM-DD)
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
        search_date = pd.to_datetime(target_date)
    except ValueError:
        print("\n[BŁĄD] Nieprawidłowy format daty.")
        print("Użyj formatu RRRR-MM-DD, np. 2024-07-15")
        sys.exit(1)

    # 3. Wyszukanie rezerwacji na wskazany dzień przyjazdu
    # Dataframe używa kolumny arrival_date jako daty zameldowania w hotelu.
    day_bookings = df[df['arrival_date'] == search_date]

    if day_bookings.empty:
        print(f"\n[INFO] W wygenerowanym zbiorze nie wylosowano żadnych rezerwacji na dzień: {target_date}.")
        sys.exit(0)

    print(f"\n=== ZNALEZIONO {len(day_bookings)} REZERWACJI DLA DNIA: {target_date} ===")
    print("Analiza oparta na pierwszej (przykładowej) rezerwacji z tego dnia.\n")
    
    # Wybieramy jeden (pierwszy) rekord dla przykładu
    row = day_bookings.iloc[0]

    # 4. Słownik opisowy i interpretacyjny (Mapowanie kolumn)
    # Kolumna -> (Tytuł Wyświetlany, Definicja / Opis pola biznesowego)
    field_metadata = {
        'booking_id': ("ID Rezerwacji", "Unikalny numer identyfikatora rezerwacji w systemie."),
        'arrival_date': ("Data Przyjazdu", "Zaplanowany dzień rozpoczęcia pobytu (check-in) w hotelu."),
        'lead_time': ("Wyprzedzenie (dni)", "Liczba dni pomiędzy dniem dokonania zakupu a datą przyjazdu. Większa liczba oznacza 'early-birds'."),
        'is_flexible': ("Taryfa Elastyczna", "Typ zakupionej oferty. True = można anulować bezkosztowo. False = bezzwrotna rezerwacja."),
        'actual_temp': ("Fakt. Temperatura (°C)", "Zanotowana maksymalna temperatura powietrza, jaka faktycznie panowała w tym dniu."),
        'actual_precip': ("Fakt. Opady (mm)", "Zanotowana suma opadów deszczu/śniegu dla tego dnia."),
        'actual_wind': ("Fakt. Wiatr (km/h)", "Zanotowana maksymalna prędkość wiatru tego dnia."),
        'forecast_14d_temp': ("Prognoza Temp. (T-14)", "Temperatura jakiej spodziewano się na całe 14 dni przed datą przyjazdu."),
        'forecast_14d_precip': ("Prognoza Opadów(T-14)", "Opady atmosferyczne prognozowane na 14 dni przed datą przyjazdu."),
        'forecast_3d_temp': ("Prognoza Temp. (T-3)", "Zaktualizowana prognoza temp. wydana na 3 dni przed (moment tzw. 'pakowania walizek')."),
        'forecast_3d_precip': ("Prognoza Opadów(T-3)", "Zaktualizowana prognoza opadów wydana na 3 dni przed przyjazdem."),
        'weather_penalty_14d': ("Kara Pogod. (T-14)", "Wyliczony naukowo wskaźnik 'złej pogody' ze starej prognozy (T-14). Ulewy zwiększają go szybciej niż chłód."),
        'weather_penalty_3d': ("Kara Pogod. (T-3)", "Wskaźnik 'złej pogody' z nowej prognozy (T-3) – tj. to, co klient widzi w Appce pogodowej tuż przed podróżą."),
        'weather_shock': ("Szok Pogodowy", "Rożnica między złą prognozą T-3 a T-14. Skok w górę >0 oznacza rozczarowanie klienta, bo pogoda się zepsuła."),
        'cancellation_prob': ("P-stwo Anulacji", "Zamodelowane algorytmem Sigmoidy ryzyko odwołania przyjazdu przez tego klienta (0.0 to 0%, 1.0 to 100%)."),
        'is_canceled': ("Status Anulacji", "Faktyczny ostateczny wynik symulacji zdarzenia dla gościa w systemie operacyjnym (0 lub 1).")
    }

    # Formatowanie Wyjścia w Konsoli
    print("-" * 125)
    print(f"{'KOLUMNA (Nazwa Zmiennej)':<25} | {'WART. ODCZYTANA':<17} | {'INTERPRETACJA & OPIS BIZNESOWY'}")
    print("-" * 125)
    
    for col_name, (display_name, definition) in field_metadata.items():
        if col_name in row:
            raw_val = row[col_name]
            
            # Formatowanie Wartości:
            if isinstance(raw_val, float):
                val_str = f"{raw_val:.2f}"
            else:
                val_str = str(raw_val)
                
            # Poprawa wizualna daty
            if col_name == 'arrival_date':
                val_str = val_str.split(' ')[0]
            
            # Dynamiczny Kontekst Biznesowy powiązany z WARTOŚCIĄ odczytaną rekordu:
            dynamic_context = f"{definition}"
            
            # - Taryfa -
            if col_name == 'is_flexible':
                dynamic_context += " ➔ Gość WYBRAŁ opcję bezstresową (FLEX)." if raw_val else " ➔ Gość MA Taryfę sztywną (Non-Ref)."
                
            # - Szok Pogodowy - 
            elif col_name == 'weather_shock':
                if raw_val > 4.0:
                    dynamic_context += f" ➔ KRYTYCZNE ROZCZAROWANIE (Szok > 4.0)! Prognozy pogorszyły się drastycznie na dniach przed przyjazdem."
                elif raw_val > 0.5:
                    dynamic_context += f" ➔ Lekki zawód. Było ciepło w T-14, zrobiło się gorzej w T-3."
                else:
                    dynamic_context += f" ➔ Pokój ducha. Pogoda jest podobna lub lepsza niż się spodziewano (Brak Szoku)."
                    
            # - Anulacja -
            elif col_name == 'is_canceled':
                if raw_val == 1:
                    dynamic_context += " ➔ WYNIK: Klient ZREZYGNOWAŁ i anulował pobyt (CHURN)."
                else:
                    dynamic_context += " ➔ WYNIK: Klient PRZYJECHAŁ zgodnie z planem."
                    
            # -  Prawdopodobieństwo churnu
            elif col_name == 'cancellation_prob':
                prob_pct = raw_val * 100
                if prob_pct > 30:
                    dynamic_context += f" ➔ Ryzyko było bardzo WYSOKIE ({prob_pct:.1f}%)."
                elif prob_pct >= 5:
                    dynamic_context += f" ➔ Ryzyko było standardowe ({prob_pct:.1f}%)."

            print(f"{col_name:<25} | {val_str:<17} | {dynamic_context}")
            
    print("-" * 125)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Statystyki i deskrypcja pojedynczego dnia z modelu hotelowego.')
    parser.add_argument('-d', '--date', type=str, help='Data wejściowa format: RRRR-MM-DD (np. 2024-07-15)')
    
    args = parser.parse_args()
    
    FILE_PKL = 'data/final_hotel_dataset.pkl'
    
    if args.date:
        interpret_day(FILE_PKL, args.date)
    else:
        print("💡 Nie przekazano parametru daty w konsoli.")
        user_input = input("⏳ Podaj interesującą Cię datę do analizy (format: RRRR-MM-DD), np. 2024-08-15: ").strip()
        if user_input:
            interpret_day(FILE_PKL, user_input)
        else:
            print("Zatrzymano skrypt.")
