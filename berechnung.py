import pandas as pd
import numpy as np

df = pd.read_csv("Data/precomputed_route_insights.csv")

# Filter, damit keine 0 oder NaN berücksichtigt werden
valid = (df['mae_holt'] > 0) & (df['mae_sarima'] > 0) & \
        (~df['mae_holt'].isna()) & (~df['mae_sarima'].isna())

df_valid = df[valid]

holt_besser = (df_valid['mae_holt'] < df_valid['mae_sarima']).sum()
sarima_besser = (df_valid['mae_sarima'] < df_valid['mae_holt']).sum()
gleich = (df_valid['mae_holt'] == df_valid['mae_sarima']).sum()

print(f"Holt-Winters besser in {holt_besser} Fällen")
print(f"SARIMA besser in {sarima_besser} Fällen")
print(f"Fehler gleich in {gleich} Fällen")



data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv", dtype={14: str})
print(data.columns.tolist())
# Gruppieren nach UNIQUE_CARRIER_NAME und zählen, wie viele unterschiedliche UNIQUE_CARRIER_ENTITY pro Airline existieren
counts = data.groupby("UNIQUE_CARRIER_NAME")["UNIQUE_CARRIER_ENTITY"].nunique()

# Airlines mit mehr als einer Entity herausfiltern
mehrere_entities = counts[counts > 1]

print(mehrere_entities)

# Gruppierung nach Route + Carrier Name
grouped = data.groupby(["ORIGIN", "DEST", "UNIQUE_CARRIER_NAME"])["UNIQUE_CARRIER_ENTITY"].nunique().reset_index()

# Nur die Fälle mit mehr als 1 Entity (also mehrere Tochtergesellschaften)
multi_entity_routes = grouped[grouped["UNIQUE_CARRIER_ENTITY"] > 1]

# Ergebnis anschauen
print(multi_entity_routes)
