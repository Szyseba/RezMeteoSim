import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def analyze_and_plot_daily_accuracy(input_file="data/raw_weather.pkl", output_html="raport_html/daily_weather_accuracy.html"):
    print(f"Wczytywanie danych pogodowych z: {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"[BŁĄD] Plik {input_file} nie istnieje. Uruchom najpierw module_weather.py.")
        return
        
    df = pd.read_pickle(input_file)
    
    # -------------------------------------------------------------------------
    # OBLICZANIE BŁĘDÓW I TRENDU (DZIENNIE)
    # -------------------------------------------------------------------------
    # Błąd bezwzględny = |Prognoza - Faktyczna|
    df['abs_err_3d_temp'] = np.abs(df['forecast_3d_temp'] - df['actual_temp'])
    df['abs_err_3d_precip'] = np.abs(df['forecast_3d_precip'] - df['actual_precip'])
    
    df['abs_err_14d_temp'] = np.abs(df['forecast_14d_temp'] - df['actual_temp'])
    df['abs_err_14d_precip'] = np.abs(df['forecast_14d_precip'] - df['actual_precip'])
    
    # Błąd biegunowy
    df['err_3d_temp'] = df['forecast_3d_temp'] - df['actual_temp']
    df['err_3d_precip'] = df['forecast_3d_precip'] - df['actual_precip']
    df['err_14d_temp'] = df['forecast_14d_temp'] - df['actual_temp']
    df['err_14d_precip'] = df['forecast_14d_precip'] - df['actual_precip']
    
    # Średnią kroczącą z ostatnich 14 dni usuwamy na rzecz względnej różnicy (Prognoza - Faktyczna)
    
    # -------------------------------------------------------------------------
    # WSKAŹNIK DYSKOMFORTU POGODOWEGO
    # -------------------------------------------------------------------------
    def calc_discomfort(temp, precip, wind, month):
        penalty = np.zeros(len(temp))
        # Wiatr (kary za wiatr > 20 km/h)
        penalty += np.maximum(0, wind - 20) * 0.4
        # Opad bazowo
        penalty += precip * 1.5
        
        is_summer = month.isin([6, 7, 8])
        is_winter = month.isin([12, 1, 2])
        is_mid = ~is_summer & ~is_winter
        
        # Lato (upał > 30 to źle, za zimno < 20 też średnio)
        penalty += np.where(is_summer & (temp > 30), (temp - 30) * 2.5, 0)
        penalty += np.where(is_summer & (temp < 20), (20 - temp) * 1.0, 0)
        
        # Zima
        # Ciapa (0 do 3 stopni) + deszcz -> bardzo źle
        is_slop = is_winter & (temp >= 0) & (temp <= 3)
        penalty += np.where(is_slop, 5 + precip * 2.5, 0)
        # Śnieżek (poniżej zera) -> ładniej, łagodzimy karę za opady
        is_snow = is_winter & (temp < 0) & (precip > 0)
        penalty -= np.where(is_snow, precip * 1.0, 0)
        # Srogi mróz
        penalty += np.where(is_winter & (temp < -10), (-10 - temp) * 0.5, 0)
        
        # Wiosna/Jesień (wymaga po prostu w miarę umiarkowanych temperatur)
        penalty += np.where(is_mid & (temp < 10), (10 - temp) * 1.0, 0)
        
        return np.maximum(0, penalty)

    df['month'] = df['date'].dt.month
    df['discomfort_actual'] = calc_discomfort(df['actual_temp'], df['actual_precip'], df['actual_wind'], df['month'])
    df['discomfort_t3'] = calc_discomfort(df['forecast_3d_temp'], df['forecast_3d_precip'], df['actual_wind'], df['month'])
    df['discomfort_t14'] = calc_discomfort(df['forecast_14d_temp'], df['forecast_14d_precip'], df['actual_wind'], df['month'])

    print("Dane obliczone pomyślnie. Przygotowuję interaktywną wizualizację do HTML...")
    
    # -------------------------------------------------------------------------
    # WIZUALIZACJA W PLOTLY
    # -------------------------------------------------------------------------
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "1. Prognoza a Faktyczna Temperatura [°C]",
            "2. Prognoza a Faktyczne Opady [mm]",
            "3. Różnica względna (Prognoza - Faktyczna): Temperatura [°C] i Opady [mm]",
            "4. Indeks Dyskomfortu Pogodowego (Upał, Ciapa, Silny Wiatr -> wyżej = gorsza pogoda)"
        )
    )
    
    x_vals = df['date']
    
    # Wykres 1
    fig.add_trace(go.Scatter(x=x_vals, y=df['actual_temp'], name="Fakt. Temp", mode='lines', line=dict(color='gray', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['forecast_14d_temp'], name="Prognoza Temp (T-14)", mode='lines', line=dict(color='#E76F51', width=1.5, dash='dot'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['forecast_3d_temp'], name="Prognoza Temp (T-3)", mode='lines', line=dict(color='#2A9D8F', width=2)), row=1, col=1)
    
    # Wykres 2
    fig.add_trace(go.Scatter(x=x_vals, y=df['actual_precip'], name="Fakt. Opad", mode='lines', line=dict(color='gray', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['forecast_14d_precip'], name="Prognoza Opad (T-14)", mode='lines', line=dict(color='#E63946', width=1.5, dash='dot'), opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['forecast_3d_precip'], name="Prognoza Opad (T-3)", mode='lines', line=dict(color='#457B9D', width=2)), row=2, col=1)
    
    # Wykres 3
    fig.add_trace(go.Bar(x=x_vals, y=df['err_14d_temp'], name="Różnica Temp (T-14)", marker_color='#E76F51', opacity=0.6), row=3, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=df['err_3d_temp'], name="Różnica Temp (T-3)", marker_color='#2A9D8F', opacity=0.8), row=3, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=df['err_14d_precip'], name="Różnica Opad (T-14)", marker_color='#E63946', opacity=0.6), row=3, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=df['err_3d_precip'], name="Różnica Opad (T-3)", marker_color='#457B9D', opacity=0.8), row=3, col=1)

    # Linia odniesienia 0 dl Wykresu 3
    fig.add_shape(type="line", x0=x_vals.iloc[0], x1=x_vals.iloc[-1], y0=0, y1=0, line=dict(color="Gray", dash="solid"), row=3, col=1)

    # Wykres 4
    fig.add_trace(go.Scatter(x=x_vals, y=df['discomfort_actual'], name="Fakt. Dyskomfort", mode='lines', line=dict(color='gray', width=3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['discomfort_t14'], name="Dyskomfort (T-14)", mode='lines', line=dict(color='#E76F51', width=1.5, dash='dot'), opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df['discomfort_t3'], name="Dyskomfort (T-3)", mode='lines', line=dict(color='#2A9D8F', width=2)), row=4, col=1)

    fig.update_layout(
        title="Dzienny przebieg błędów prognozy pogody z wyborem zakresu dat (Cały okres analizy)",
        height=1300,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    
    # Suwak i przyciski zakresu dat na dolnej osi X (Wykres 4)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1r", step="year", stepmode="backward"),
                dict(step="all", label="Cały okres")
            ]),
            bgcolor="#f4f4f4",
            activecolor="#cccccc"
        ),
        row=4, col=1
    )
    
    fig.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Opady (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Różnica: Prognoza - Faktyczna", row=3, col=1)
    fig.update_yaxes(title_text="Indeks (0 = Idealnie)", row=4, col=1)
    
    # Zapis
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    
    print(f"✅ Zakończono z sukcesem. Wygenerowano raport analityczny: {output_html}")
    
if __name__ == "__main__":
    analyze_and_plot_daily_accuracy()
