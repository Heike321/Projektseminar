import pandas as pd

# # Load list of 473 valid connections
valid_connections = pd.read_excel("Data/Connections.xlsx")

# Load raw data files from 2022, 2023, 202
df2022 = pd.read_csv("Data/T_T100I_SEGMENT_ALL_CARRIER_2022.csv")
df2023 = pd.read_csv("Data/T_T100I_SEGMENT_ALL_CARRIER_2023.csv")
df2024 = pd.read_csv("Data/T_T100I_SEGMENT_ALL_CARRIER_2024.csv")

# Combine all yearly datasets into one DataFrame
df_all = pd.concat([df2022, df2023, df2024], ignore_index=True)

# Define the columns that identify a connection (without year/month)
connection_keys = ["AIRLINE_ID", "UNIQUE_CARRIER_ENTITY", "ORIGIN", "DEST", "AIRCRAFT_TYPE"]

# Merge: keep only rows in df_all that match a valid connection (inner join)
filtered_df = df_all.merge(valid_connections[connection_keys], how="inner", on=connection_keys)

# Group by connection + year + month and sum numeric values
grouped = filtered_df.groupby(connection_keys + ["YEAR", "MONTH"]).agg("sum").reset_index()

# Overwrite the original Connections.xlsx with grouped data
grouped.to_excel("Data/Grouped_Valid_Connections.xlsx", index=False)

# Print how many rows were kept
print(f" Raw data rows before grouping: {len(filtered_df)}")
print(f" Grouped rows saved: {len(grouped)} (expected: 17.028)")

