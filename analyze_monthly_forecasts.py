import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def analyze_and_plot_monthly_accuracy(input_file="data/raw_weather.pkl", output_html="raport_html/monthly_weather_accuracy.html"):
    print(f"Wczytywanie danych pogodowych z: {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"[BŁĄD] Plik {input_file} nie istnieje. Uruchom najpierw module_weather.py lub odpowiedni skrypt pobierający dane.")
        return
        
    df = pd.read_pickle(input_file)
    
    # Przygotowanie miesiąca w formacie YYYY-MM
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # -------------------------------------------------------------------------
    # OBLICZANIE BŁĘDÓW I TRENDU
    # -------------------------------------------------------------------------
    # Błąd = Prognoza - Faktyczna
    df['err_3d_temp'] = df['forecast_3d_temp'] - df['actual_temp']
    df['err_3d_precip'] = df['forecast_3d_precip'] - df['actual_precip']
    
    df['err_14d_temp'] = df['forecast_14d_temp'] - df['actual_temp']
    df['err_14d_precip'] = df['forecast_14d_precip'] - df['actual_precip']
    
    # Oszacowanie "gorszej pogody" w prognozie:
    # Uznajemy, że prognoza zapowiadała GORSZĄ pogodę niż była w rzeczywistości, jeżeli:
    # 1. Prognozowana temperatura była niższa od rzeczywistej.
    # 2. Prognozowane opady były większe od rzeczywistych.
    df['worse_3d_temp'] = df['err_3d_temp'] < 0
    df['worse_3d_precip'] = df['err_3d_precip'] > 0
    
    df['worse_14d_temp'] = df['err_14d_temp'] < 0
    df['worse_14d_precip'] = df['err_14d_precip'] > 0
    
    # -------------------------------------------------------------------------
    # AGREGACJA MIESIĘCZNA
    # -------------------------------------------------------------------------
    monthly_stats = df.groupby('year_month').agg(
        # Metryka RMSE (Root Mean Squared Error) do oceny wielkości błędu / skuteczności
        rmse_3d_temp=('err_3d_temp', lambda x: np.sqrt((x**2).mean())),
        rmse_14d_temp=('err_14d_temp', lambda x: np.sqrt((x**2).mean())),
        
        rmse_3d_precip=('err_3d_precip', lambda x: np.sqrt((x**2).mean())),
        rmse_14d_precip=('err_14d_precip', lambda x: np.sqrt((x**2).mean())),
        
        # Obliczenie w ilu % przypadków w danym miesiącu pesymistycznie prognozowano pogodę
        pct_worse_3d_temp=('worse_3d_temp', 'mean'),
        pct_worse_14d_temp=('worse_14d_temp', 'mean'),
        
        pct_worse_3d_precip=('worse_3d_precip', 'mean'),
        pct_worse_14d_precip=('worse_14d_precip', 'mean')
    ).reset_index()
    
    # Przeliczenie wskaźników trendu na procenty
    for col in [c for c in monthly_stats.columns if c.startswith('pct_')]:
        monthly_stats[col] = monthly_stats[col] * 100
        
    print("Dane zagregowane pomyślnie. Przygotowuję wizualizację do HTML...")
    
    # -------------------------------------------------------------------------
    # WIZUALIZACJA W PLOTLY
    # -------------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Skuteczność prognoz: RMSE dla Temperatury [°C] (niższy = lepszy)",
            "2. Skuteczność prognoz: RMSE dla Opadów [mm] (niższy = lepszy)",
            "3. Trend pesymistyczny: W ilu % przypadków prognoza zapowiadała gorszą pogodę niż była? (linia 50% = brak trendu)"
        )
    )
    
    x_vals = monthly_stats['year_month']
    
    # Wykres 1: Skuteczność T-3 vs T-14 (Temperatura)
    fig.add_trace(go.Bar(x=x_vals, y=monthly_stats['rmse_3d_temp'], name="Błąd RMSE (T-3) Temp", marker_color='#2A9D8F'), row=1, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=monthly_stats['rmse_14d_temp'], name="Błąd RMSE (T-14) Temp", marker_color='#E76F51'), row=1, col=1)
    
    # Wykres 2: Skuteczność T-3 vs T-14 (Opady)
    fig.add_trace(go.Bar(x=x_vals, y=monthly_stats['rmse_3d_precip'], name="Błąd RMSE (T-3) Opad", marker_color='#457B9D'), row=2, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=monthly_stats['rmse_14d_precip'], name="Błąd RMSE (T-14) Opad", marker_color='#E63946'), row=2, col=1)
    
    # Wykres 3: Analiza trendu, czy częściej zapowiadano gorzej niż było
    fig.add_trace(go.Scatter(x=x_vals, y=monthly_stats['pct_worse_14d_temp'], name="% Zimniej wg T-14", mode='lines+markers', line=dict(color='#E76F51', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=monthly_stats['pct_worse_3d_temp'], name="% Zimniej wg T-3", mode='lines+markers', line=dict(color='#2A9D8F', dash='solid')), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=x_vals, y=monthly_stats['pct_worse_14d_precip'], name="% Więcej deszczu wg T-14", mode='lines+markers', line=dict(color='#E63946', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=monthly_stats['pct_worse_3d_precip'], name="% Więcej deszczu wg T-3", mode='lines+markers', line=dict(color='#457B9D', dash='solid')), row=3, col=1)

    # Globalna linia odniesienia 50%
    fig.add_shape(type="line", x0=x_vals.iloc[0], x1=x_vals.iloc[-1], y0=50, y1=50, line=dict(color="Gray", dash="dot"), row=3, col=1)

    fig.update_layout(
        title="Kompleksowa analiza trafności prognoz (14-dniowych i 3-dniowych) wg miesięcy",
        height=1000,
        hovermode="x unified",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    
    # Dodanie podpisów osi
    fig.update_yaxes(title_text="Błąd (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Błąd (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Częstotliwość (%)", row=3, col=1)
    
    # Zapis
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    
    print(f"✅ Zakończono z sukcesem. Wygenerowano raport analityczny: {output_html}")
    
    # Obliczenie kilku agregatów do logów
    print("\n--- PODSUMOWANIE STATYSTYCZNE ---")
    print(f"Średni błąd (RMSE) w temperaturze: T-3 = {monthly_stats['rmse_3d_temp'].mean():.2f}°C, T-14 = {monthly_stats['rmse_14d_temp'].mean():.2f}°C")
    print(f"Średni błąd (RMSE) w opadach:      T-3 = {monthly_stats['rmse_3d_precip'].mean():.2f} mm, T-14 = {monthly_stats['rmse_14d_precip'].mean():.2f} mm")
    
    avg_pes_t14 = monthly_stats['pct_worse_14d_temp'].mean()
    avg_pes_t3 = monthly_stats['pct_worse_3d_temp'].mean()
    print(f"\nTrend (Czy prognozowano zimniej niż faktycznie było?):")
    print(f"- W T-14: średnio w {avg_pes_t14:.1f}% dni w miesiącu.")
    print(f"- W T-3:  średnio w {avg_pes_t3:.1f}% dni w miesiącu.")

if __name__ == "__main__":
    analyze_and_plot_monthly_accuracy()
