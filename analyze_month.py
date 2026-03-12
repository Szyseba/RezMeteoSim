import pandas as pd
import plotly.graph_objects as go
import os
import argparse
import sys
import numpy as np
from datetime import datetime

def analyze_month(input_file: str, target_year: int, target_month: int, output_dir: str):
    """
    Analizuje podany miesiąc (np. 7) w Konkretnym Roku (np. 2024).
    Pozwala to na zbadanie "niefiltrowanej" badawczo i niewygładzonej korelacji 
    szeregów czasowych opartych na unikalnych i prawdziwych zdarzeniach pogodowych.
    """
    print(f"[{os.path.basename(__file__)}] Wczytywanie danych: {input_file}")
    try:
        df = pd.read_pickle(input_file)
    except FileNotFoundError:
        print(f"[BŁĄD] Nie znaleziono pliku '{input_file}'. Uruchom najpierw run_pipeline.py")
        sys.exit(1)
        
    if target_month < 1 or target_month > 12:
        print("[BŁĄD] Miesiąc musi być liczbą od 1 do 12.")
        sys.exit(1)
        
    MONTH_NAMES = ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", 
                   "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"]
    month_name = MONTH_NAMES[target_month - 1]
    
    print(f"=== ANALIZA MIESIĄCA: {month_name.upper()} {target_year} ===")
    
    # 1. Filtrowanie danych do podanego RRRR-MM
    mask = (df['arrival_date'].dt.year == target_year) & (df['arrival_date'].dt.month == target_month)
    df_month = df[mask].copy()
    
    if df_month.empty:
        print(f"[INFO] Brak wygenerowanych rezerwacji dla daty: {target_year}-{target_month:02d}.")
        sys.exit(0)
        
    # 2. Agregacja dzienna dla Konkretnego Roku (Timeline)
    # Zostawiamy unikalne daty na osi czasu
    df_month['day_date'] = df_month['arrival_date'].dt.date
    
    daily_stats = df_month.groupby('day_date').agg(
        total_bookings=('booking_id', 'count'),
        canceled_bookings=('is_canceled', 'sum'),
        avg_temp=('actual_temp', 'mean'),
        avg_penalty_3d=('weather_penalty_3d', 'mean')
    ).reset_index()
    
    daily_stats['realized_bookings'] = daily_stats['total_bookings'] - daily_stats['canceled_bookings']
    
    # 3. Weryfikacja Korelacji (Matematycznie) na Ciągłym Szeregu Czasowym
    # Badamy współczynnik Pearsona między złą pogodą a liczbą anulacji na konkretny dzień 
    correlation = daily_stats['avg_penalty_3d'].corr(daily_stats['canceled_bookings'])
    
    print("\n--- STATYSTYKI KORELACJI ---")
    print(f"Współczynnik korelacji (Pearson) dla Wskaźnika Złej Pogody (T-3) i Anulacji: {correlation:.3f}")
    if correlation > 0.5:
        print("💡 Silna pozytywna korelacja: Fatalna pogoda napędzała masowe anulacje w tym miesiącu.")
    elif correlation > 0.2:
        print("💡 Umiarkowana korelacja: Wyczuwalny wpływ pogorszonej pogody na obłożenie dni.")
    elif correlation > -0.2:
        print("💡 Brak istotnej korelacji: Anulacje w tym miesiącu zdarzały się dość losowo (brak skrajnych anomalii).")
    else:
        print("💡 Odwrotna korelacja: Nietypowe zjawisko statystyczne w danych.")

    # 4. Generowanie Interaktywnego Wykresu HTML
    os.makedirs(output_dir, exist_ok=True)
    html_filename = f"analiza_miesiac_{target_year}_{target_month:02d}.html"
    filepath = os.path.join(output_dir, html_filename)
    
    fig = go.Figure()

    # Słupki skumulowane - Zrealizowane
    fig.add_trace(go.Bar(
        x=daily_stats['day_date'],
        y=daily_stats['realized_bookings'],
        name='Zrealizowane Pobyty',
        marker_color='#A8DADC',
        yaxis='y'
    ))

    # Słupki skumulowane - Anulowane
    fig.add_trace(go.Bar(
        x=daily_stats['day_date'],
        y=daily_stats['canceled_bookings'],
        name='Anulowane',
        marker_color='#F4A261',
        yaxis='y'
    ))

    # Linia ciągła - Faktyczna Temperatura (Oś Prawa)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['avg_temp'],
        name='Rzeczywista Temp (°C)',
        mode='lines+markers',
        line=dict(color='#E9C46A', width=2),
        yaxis='y2'
    ))
    
    # Linia kropkowana - Wskaźnik Złej Pogody (Oś Prawa)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['avg_penalty_3d'],
        name='Wskaźnik Złej Pogody (T-3)',
        mode='lines+markers',
        line=dict(color='#E76F51', width=3, dash='dash'),
        yaxis='y2'
    ))

    # Formatowanie Wykresu
    fig.update_layout(
        title=f"Wyniki Rezerwacyjne: {month_name} {target_year}<br><sup>Korelacja Zła Pogoda a Anulacje: r = {correlation:.2f}</sup>",
        xaxis_title=f"Data Przyjazdu",
        xaxis=dict(
            tickmode='linear',
            dtick="86400000",   # 1 Day in Milliseconds (Plotly axis tick)
            tickformat="%d %b"  # Formatuje datę, np. '15 Jul'
        ), 
        yaxis=dict(title="Liczba Rezerwacji", side="left", showgrid=False),
        yaxis2=dict(
            title="Wartość Wskaźnika / Temperatura", 
            side="right", 
            overlaying="y", 
            showgrid=False
        ),
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    fig.write_html(filepath)
    print(f"\n✅ ZAKOŃCZONO. Wygenerowano nowy widok HTML w pliku: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analiza obłożenia i szoku dla Konkretnego Miesiąca w Roku.')
    parser.add_argument('-d', '--date', type=str, help='Miesiąc i Rok w formacie YYYY-MM (np. 2024-07 dla Lipca 2024)')
    
    args = parser.parse_args()
    
    INPUT_PKL = 'data/final_hotel_dataset.pkl'
    OUTPUT_FOLDER = 'raport_html'
    
    if args.date is not None:
        try:
            year, month = map(int, args.date.split('-'))
            analyze_month(INPUT_PKL, year, month, OUTPUT_FOLDER)
        except ValueError:
            print("[BŁĄD] Użyj formatu RRRR-MM (np. 2024-07)")
    else:
        print("💡 Nie przekazano parametru daty w konsoli.")
        try:
            m_input = input("⏳ Wpisz RRRR-MM do analizy (np. 2024-07): ").strip()
            year, month = map(int, m_input.split('-'))
            analyze_month(INPUT_PKL, year, month, OUTPUT_FOLDER)
        except ValueError:
            print("[BŁĄD] Należy podać rok i miesiąc oddzielone myślnikiem (np. 2024-07).")
