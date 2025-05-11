import pandas as pd
import json
import math

# Function to create a unique key for each connection
def make_key(air, uce, org, dst, atp):
    return f"{air}-{uce}-{org}-{dst}-{atp}"

def preprocess():
    # List of raw flight connection data files
    files = [
        "Data/T_T100I_SEGMENT_ALL_CARRIER_2022.csv",
        "Data/T_T100I_SEGMENT_ALL_CARRIER_2023.csv",
        "Data/T_T100I_SEGMENT_ALL_CARRIER_2024.csv"
    ]

    #Load Excel file with valid connections
    df_connections = pd.read_excel("Data/Connections.xlsx")
    
    # Convert the valid connections into a set of keys for easier lookup
    passed = [tuple(row) for row in df_connections.values]
    passed_keys = {make_key(*con): con for con in passed}

    # Prepare an empty list to hold all processed dataframes
    all_grouped = []
    
    # Iterate over each year file and process the data
    for f, year in zip(files, [2022, 2023, 2024]):
        df = pd.read_csv(f, usecols=[
            "PASSENGERS", "DEPARTURES_PERFORMED", "SEATS",
            "AIRLINE_ID", "UNIQUE_CARRIER_ENTITY", "ORIGIN",
            "DEST", "AIRCRAFT_TYPE", "MONTH"
        ])
        
        # Create a unique connection key for each row
        df["con_key"] = df.apply(lambda row: make_key(
            row["AIRLINE_ID"],
            row["UNIQUE_CARRIER_ENTITY"],
            row["ORIGIN"],
            row["DEST"],
            row["AIRCRAFT_TYPE"]
        ), axis=1)
        
        # Filter the dataframe to include only valid connections
        filtered = df[df["con_key"].isin(passed_keys)]
        
        # Group the filtered data by connection key and month, then aggregate the numeric values
        grouped = filtered.groupby(["con_key", "MONTH"], as_index=False).agg({
            "PASSENGERS": "sum",
            "SEATS": "sum",
            "DEPARTURES_PERFORMED": "sum",
            "AIRLINE_ID": "first",
            "ORIGIN": "first",
            "DEST": "first"
        })
        
        # Calculate Average Passengers per Flight (rounded up)
        grouped["AVG_PAX_PER_FLIGHT"] = grouped.apply(
            lambda row: math.ceil(row["PASSENGERS"] / row["DEPARTURES_PERFORMED"]) if row["DEPARTURES_PERFORMED"] > 0 else 0,
            axis=1
        )
        
        # Calculate Load Factor (passengers divided by seats)
        grouped["LOAD_FACTOR"] = grouped.apply(
            lambda row: row["PASSENGERS"] / row["SEATS"] if row["SEATS"] > 0 else 0,
            axis=1
        )

        # Add the year column to the grouped data
        grouped["YEAR"] = year
        
        # Append the grouped data for the current year to the list
        all_grouped.append(grouped)

    # Concatenate all years' grouped data into a single DataFrame
    final_grouped = pd.concat(all_grouped, ignore_index=True)
    
    # Save the final grouped data into a single CSV file
    final_grouped.to_csv("Data/Grouped_All_Valid_Connections.csv", index=False)

    # Print the summary of processed rows
    print(f"Total filtered and grouped rows saved: {len(final_grouped)}")

    # 
    airports_df = pd.read_csv("airports.dat", header=None, names=[
        "Airport_ID", "Name", "City", "Country", "IATA", "ICAO",
        "Latitude", "Longitude", "Altitude", "Timezone", "DST",
        "Tz_database_time_zone", "Type", "Source"
    ])
    
    # Create a dictionary to map IATA codes to airport names
    iata_to_name = dict(zip(airports_df["IATA"], airports_df["Name"]))

    # Sample a dataframe from the first year to get the connection labels
    sample_df = all_grouped[0][all_grouped[0]["con_key"].isin(passed_keys)]
    
    # Group by connection key to create a label for each origin-destination pair
    grouped_labels = sample_df.groupby("con_key").agg({
        "ORIGIN": "first",
        "DEST": "first"
    }).reset_index()

    # Create entries for the dropdown menu
    dropdown_entries = []
    for _, row in grouped_labels.iterrows():
        origin = row["ORIGIN"]
        dest = row["DEST"]
        label = f"{iata_to_name.get(origin, origin)} ({origin}) → {iata_to_name.get(dest, dest)} ({dest})"
        dropdown_entries.append({
            "label": label,
            "value": row["con_key"]
        })

    # Save the dropdown entries as a JSON file for use in the dashboard
    with open("Data/valid_connections.json", "w") as f:
        json.dump(dropdown_entries, f)

if __name__ == "__main__":
    preprocess()






