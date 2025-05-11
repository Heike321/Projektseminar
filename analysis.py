import pandas as pd

df = pd.concat([pd.read_csv(f"processed_{y}.csv") for y in [2022, 2023, 2024]])


def compute_top_routes(df, top_n=10):
    # Zerlege Verbindungsschlüssel, falls nötig
    parts = df['con_key'].str.split('-', expand=True)
    df['ORIGIN'] = parts[2]
    df['DEST'] = parts[3]

    # 1. Zuerst auf Monatsbasis aggregieren (alle Airlines/Aircrafts werden zusammengefasst)
    monthly_routes = df.groupby(['YEAR', 'MONTH', 'ORIGIN', 'DEST'])['PASSENGERS'].sum().reset_index()

    # 2. Dann auf Jahresbasis summieren
    total_routes = monthly_routes.groupby(['ORIGIN', 'DEST'])['PASSENGERS'].sum().reset_index()

    # 3. Route-Label erzeugen
    total_routes['ROUTE'] = total_routes['ORIGIN'] + " → " + total_routes['DEST']

    return total_routes.sort_values(by='PASSENGERS', ascending=False).head(top_n)

def compute_top_airlines(df, top_n=10):
    # Split the connection key to extract the airline identifier (if needed)
    parts = df['con_key'].str.split('-', expand=True)
    df['AIRLINE_ID'] = parts[0]

    # Step 1: Aggregate monthly data per airline (to avoid double-counting across different aircraft/routes)
    monthly_airline_data = df.groupby(['YEAR', 'MONTH', 'AIRLINE_ID'])['PASSENGERS'].sum().reset_index()

    # Step 2: Sum total passengers per airline across all months and years
    total_airline_passengers = monthly_airline_data.groupby('AIRLINE_ID')['PASSENGERS'].sum().reset_index()

    # Step 3: Sort and return top N airlines
    return total_airline_passengers.sort_values(by='PASSENGERS', ascending=False).head(top_n)


#print(df.groupby(['ORIGIN', 'DEST', 'MONTH', 'YEAR']).size().sort_values(ascending=False).head(10))

