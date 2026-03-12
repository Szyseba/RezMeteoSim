import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def create_visualizations(input_file: str, output_dir: str):
    """
    Tworzy interaktywne wizualizacje HTML na podstawie danych wygenerowanych w pliku .pkl.
    Wymaga bibliotek: pandas, plotly.
    """
    print(f"[{os.path.basename(__file__)}] Wczytywanie danych: {input_file}")
    df = pd.read_pickle(input_file)
    
    # Utworzenie folderu na wykresy, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{os.path.basename(__file__)}] Wykresy zostaną zapisane w: {os.path.abspath(output_dir)}\n")

    # --- Wykres 1: Analiza Churn'u wg Taryfy (Flex vs Non-Flex) ---
    print("Generowanie: status_anulacji_taryfy.html ...")
    churn_rates = df.groupby('is_flexible')['is_canceled'].value_counts(normalize=True).unstack() * 100
    churn_rates.index = ['Sztywna (Non-Flex)', 'Elastyczna (Flex)']
    churn_rates.columns = ['Zrealizowana', 'Anulowana']
    churn_rates = churn_rates.reset_index()

    fig1 = px.bar(
        churn_rates, 
        x='index', 
        y=['Zrealizowana', 'Anulowana'], 
        title="Odsetek Zrealizowanych vs Anulowanych Rezerwacji wg Typu Taryfy",
        labels={'value': 'Procent (%)', 'index': 'Typ Taryfy', 'variable': 'Status'},
        barmode='stack',
        color_discrete_map={'Zrealizowana': '#A8DADC', 'Anulowana': '#F4A261'}
    )
    fig1.update_layout(yaxis=dict(range=[0, 100]))
    fig1.write_html(os.path.join(output_dir, "status_anulacji_taryfy.html"))

    # --- Wykres 2: Rozkład Szoku Pogodowego (Tylko dla Anulowanych vs Niezrealizowanych) ---
    print("Generowanie: dystrybucja_szoku.html ...")
    fig2 = px.histogram(
        df, 
        x='weather_shock', 
        color=df['is_canceled'].replace({0: 'Zrealizowana', 1: 'Anulowana'}),
        title="Dystrybucja 'Szoku Pogodowego' ze względu na Status Rezerwacji",
        labels={'weather_shock': 'Index Szoku Pogodowego (Więcej = Gorzej)', 'color': 'Status Rezerwacji'},
        nbins=50,
        barmode='overlay',
        histnorm='probability',
        color_discrete_map={'Zrealizowana': '#A8DADC', 'Anulowana': '#F4A261'}
    )
    # Dodajemy linię wskazującą nasz `shock_threshold` (4.0) od którego rosło prawdopodobieństwo anulacji
    fig2.add_vline(x=4.0, line_dash="dash", line_color="#E76F51", annotation_text="Próg krytyczny")
    fig2.write_html(os.path.join(output_dir, "dystrybucja_szoku.html"))
    
    # --- Wykres 3: Korelacja Prawdopodobieństwa Anulacji i Kary Pogodowej (Heatmap 2D / Hexbin) ---
    print("Generowanie: korelacja_kary_anulacji.html ...")
    fig3 = px.density_heatmap(
        df, 
        x="weather_penalty_3d", 
        y="cancellation_prob", 
        title="Zależność prognozowanej matematycznej szansy na Anulację od P-Score (Kary) w T-3",
        labels={
            'weather_penalty_3d': 'Oczekiwana Kara Pogodowa na 3 Dni przed', 
            'cancellation_prob': 'Wygenerowane Prawdopod. Anulacji (%)'
        },
        color_continuous_scale="PuBuGn" # Pastelowa morska zieleń -> błękit
    )
    fig3.write_html(os.path.join(output_dir, "korelacja_kary_anulacji.html"))
    
    # --- Wykres 4: Serie Czasowe (Zależność Pogody i Przyjazdów w Czasie) ---
    # Grupujemy dane per miesiąc (przyjazd)
    print("Generowanie: oblozenie_vs_pogoda_miesiecznie.html ...")
    df['arrival_month'] = df['arrival_date'].dt.to_period('M')
    monthly_data = df.groupby('arrival_month').agg(
        total_bookings=('booking_id', 'count'),
        canceled_bookings=('is_canceled', 'sum'),
        avg_temp=('actual_temp', 'mean'),
        avg_precip=('actual_precip', 'mean')
    ).reset_index()
    monthly_data['arrival_month'] = monthly_data['arrival_month'].dt.to_timestamp()
    monthly_data['realized_bookings'] = monthly_data['total_bookings'] - monthly_data['canceled_bookings']
    
    fig4 = go.Figure()
    
    # Słupki (Zrealizowane)
    fig4.add_trace(go.Bar(
        x=monthly_data['arrival_month'],
        y=monthly_data['realized_bookings'],
        name='Zrealizowane Pobyt',
        marker_color='#A8DADC',  # Pastel morski / błękitny
        yaxis='y'
    ))

    # Słupki (Anulowane) 
    fig4.add_trace(go.Bar(
        x=monthly_data['arrival_month'],
        y=monthly_data['canceled_bookings'],
        name='Anulowane',
        marker_color='#F4A261',  # Łagodny pomarańcz
        yaxis='y'
    ))
    
    # Linia (Temperatura) na Drugiej osi Y
    fig4.add_trace(go.Scatter(
        x=monthly_data['arrival_month'],
        y=monthly_data['avg_temp'],
        name='Średnia Max Temp (°C)',
        mode='lines+markers',
        line=dict(color='#E9C46A', width=3),  # Pastelowa miodowa zółć
        yaxis='y2'
    ))

    fig4.update_layout(
        title="Trendy Obłożenia i Skala Anulacji na Tle Faktycznych Temperatur",
        xaxis_title="Miesiąc Przyjazdu",
        yaxis=dict(title="Liczba Rezerwacji", side="left", showgrid=False),
        yaxis2=dict(title="Temperatura (°C)", side="right", overlaying="y", showgrid=False),
        barmode='stack',  # nakładamy anulowane na zrealizowane, by widzieć całość ("total") per miesiąc
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    fig4.write_html(os.path.join(output_dir, "oblozenie_vs_pogoda_miesiecznie.html"))

    print("\nZAKOŃCZONO. Wygenerowano wszystkie pliki HTML.")

def main():
    INPUT_FILE = "data/final_hotel_dataset.pkl"
    OUTPUT_DIR = "raport_html"
    
    if not os.path.exists(INPUT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku '{INPUT_FILE}'. Uruchom najpierw run_pipeline.py")
        return
        
    create_visualizations(INPUT_FILE, OUTPUT_DIR)

if __name__ == "__main__":
    main()
