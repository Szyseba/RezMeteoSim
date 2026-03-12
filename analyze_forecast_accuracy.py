import pandas as pd
import plotly.graph_objects as go
import os
import argparse
import sys
import numpy as np

def analyze_forecast_accuracy(input_file: str, target_year: int, target_month: int, output_dir: str):
    """
    Analizuje podany miesiąc (rok-miesiąc) pod kątem rozbieżności między:
    Prognozą długoterminową (T-14),
    Prognozą krótkoterminową (T-3),
    A stanem faktycznym w warunkach docelowych obiektu.
    Szuka dni "istotnie nietrafionych".
    """
    print(f"[{os.path.basename(__file__)}] Wczytywanie danych: {input_file}")
    try:
        df = pd.read_pickle(input_file)
    except FileNotFoundError:
        print(f"[BŁĄD] Nie znaleziono pliku '{input_file}'. Uruchom najpierw run_pipeline.py")
        sys.exit(1)
        
    MONTH_NAMES = ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", 
                   "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"]
    month_name = MONTH_NAMES[target_month - 1]
    
    print(f"=== ANALIZA DOKŁADNOŚCI PROGNOZY: {month_name.upper()} {target_year} ===")
    
    # 1. Filtrowanie danych do podanego RRRR-MM
    mask = (df['arrival_date'].dt.year == target_year) & (df['arrival_date'].dt.month == target_month)
    df_month = df[mask].copy()
    
    if df_month.empty:
        print(f"[INFO] Brak wygenerowanych rezerwacji dla daty: {target_year}-{target_month:02d}.")
        sys.exit(0)
        
    # 2. Agregowane odczucia klimatyczne z metryką P-Score (Kary Pogodowej)
    # Zamiast samej temperatury, mierzymy 'Istotne Pudło' w ogólnym odczuciu P-Score
    df_month['day_date'] = df_month['arrival_date'].dt.date
    
    # Przeliczamy referencyjny "Faktyczny P-Score" po przyjeździe na tej samej wadze (w1=2.0 deszcz, w2=0.5 dla T<22)
    w1, w2 = 2.0, 0.5
    df_month['actual_penalty'] = w1 * df_month['actual_precip'] + w2 * np.maximum(0, 22 - df_month['actual_temp'])
    
    daily_stats = df_month.groupby('day_date').agg(
        p_t14=('weather_penalty_14d', 'mean'),
        p_t3=('weather_penalty_3d', 'mean'),
        p_actual=('actual_penalty', 'mean'),
        actual_temp=('actual_temp', 'mean'),
        actual_precip=('actual_precip', 'mean')
    ).reset_index()
    
    # Znalezienie wpadki: Gdy faktyczna pogoda okazała się dramatycznie inna
    # np. T-14 i T-3 obiecywały znośny wypad (P <= 2.0), a na miejscu było leje jak z cebra (P > 5.0)
    # ALBO T-14 i T-3 zapowiadały tragedię (P > 5.0), a było pięknie słońce (P <= 1.0).
    # Jako metrykę 'Miss' bierzemy absolutny błąd między najświeższą prognozą (T-3) a Faktem (Actual) wyższy od zadanego marginesu błędu Modelu
    ERROR_MARGIN_PSCORE = 4.0 # 4 punkty na skali P-Score to zazwyczaj potężna ulewa wzięta znikąd
    
    daily_stats['forecast_error'] = np.abs(daily_stats['p_actual'] - daily_stats['p_t3'])
    daily_stats['is_miss'] = daily_stats['forecast_error'] > ERROR_MARGIN_PSCORE
    
    miss_count = daily_stats['is_miss'].sum()
    print(f"Znaleziono {miss_count} dni w których odnotowano tzw. 'Istotne Pudło' (Błąd względem T-3 > {ERROR_MARGIN_PSCORE} pkt karny).")

    # 3. Generowanie Wykresu Porównawczego HTML (Plotly)
    os.makedirs(output_dir, exist_ok=True)
    html_filename = f"trafnosc_prognozy_{target_year}_{target_month:02d}.html"
    filepath = os.path.join(output_dir, html_filename)
    
    fig = go.Figure()

    # Długoterminowa T-14 (Oś P-Score)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['p_t14'],
        name='Prognoza: 14 Dni przed (T-14)',
        mode='lines',
        line=dict(color='#A8DADC', width=2, dash='dot'), # Pastelowy Błękit
        opacity=0.6,
        yaxis='y'
    ))

    # Krótkoterminowa T-3 (Oś P-Score)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['p_t3'],
        name='Prognoza: 3 Dni przed (T-3)',
        mode='lines',
        line=dict(color='#F4A261', width=3, dash='dash'), # Pastelowy Pomarańcz
        yaxis='y'
    ))

    # Wartość Faktyczna P-Score (Oś P-Score)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['p_actual'],
        name='Pogoda Faktyczna Wsk. (Na miejscu)',
        mode='lines+markers', # Punkty zaznaczają dokładne stacje dzienne
        line=dict(color='#2A9D8F', width=4), # Solidna zieleń morska
        yaxis='y'
    ))
    
    # [NOWE] Faktyczne Opady Deszczu (Oś Parametrów / Y2)
    fig.add_trace(go.Bar(
        x=daily_stats['day_date'],
        y=daily_stats['actual_precip'],
        name='Fakt. Opad (mm)',
        marker_color='#457b9d', # Przygaszony niebieski dla deszczu
        opacity=0.3, # Półprzezroczyste, żeby nie zasłaniały linii predykcyjnych
        yaxis='y2'
    ))

    # [NOWE] Faktyczna Temperatura (Oś Parametrów / Y2)
    fig.add_trace(go.Scatter(
        x=daily_stats['day_date'],
        y=daily_stats['actual_temp'],
        name='Fakt. Temperatura (°C)',
        mode='lines',
        line=dict(color='#e9c46a', width=2), # Pastelowa Żółć
        opacity=0.7,
        yaxis='y2'
    ))
    
    # Dodanie Markerów Błędu w "Miejscach Pudeł" nałożone na krzywą faktu
    miss_points_df = daily_stats[daily_stats['is_miss']].copy()
    
    if not miss_points_df.empty:
        fig.add_trace(go.Scatter(
            x=miss_points_df['day_date'],
            y=miss_points_df['p_actual'],
            name='🚨 Istotnie Nietrafiona Prognoza',
            mode='markers',
            marker=dict(
                size=12,
                color='#E76F51', # Ostry Rdzawy/Czerwony
                symbol='x',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            # Hover text pokazujący dlaczego
            text=[f"Błąd pomiarowy: {err:.1f} Pkt karny" for err in miss_points_df['forecast_error']],
            hoverinfo="x+y+text",
            yaxis='y'
        ))

    # Formatowanie Wykresu
    fig.update_layout(
        title=f"Wiarygodność Prognoz Meteo (T-14 vs T-3 vs Fakt) – {month_name} {target_year}<br><sup>Wymiarowanie punktowe 'P-Score' (Im wyżej na osi tym gorsza pogoda)</sup>",
        xaxis_title=f"Skrzydła Dni Przyjazdów",
        xaxis=dict(
            tickmode='linear',
            dtick="86400000",   
            tickformat="%d %b"  
        ), 
        yaxis=dict(title="Wartość wskaźnika Złej Pogody (P-Score)", side="left"),
        yaxis2=dict(
            title="Wartości Fizyczne (°C / mm)", 
            side="right", 
            overlaying="y", 
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        template='plotly_white'
    )

    fig.write_html(filepath)
    print(f"\n✅ ZAKOŃCZONO. Wygenerowano nowy widok HTML z trakcjami w pliku: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analiza wiarygodności modelu predykcji pogody w Konkretnym Miesiącu.')
    parser.add_argument('-d', '--date', type=str, help='Miesiąc i Rok w formacie YYYY-MM (np. 2024-07)')
    
    args = parser.parse_args()
    
    INPUT_PKL = 'data/final_hotel_dataset.pkl'
    OUTPUT_FOLDER = 'raport_html'
    
    if args.date is not None:
        try:
            year, month = map(int, args.date.split('-'))
            analyze_forecast_accuracy(INPUT_PKL, year, month, OUTPUT_FOLDER)
        except ValueError:
            print("[BŁĄD] Użyj formatu RRRR-MM (np. 2024-07)")
    else:
        print("💡 Nie przekazano parametru daty w konsoli.")
        try:
            m_input = input("⏳ Wpisz RRRR-MM do ewaluacji prognoz (np. 2024-07): ").strip()
            year, month = map(int, m_input.split('-'))
            analyze_forecast_accuracy(INPUT_PKL, year, month, OUTPUT_FOLDER)
        except ValueError:
            print("[BŁĄD] Należy podać rok i miesiąc oddzielone myślnikiem (np. 2024-07).")
