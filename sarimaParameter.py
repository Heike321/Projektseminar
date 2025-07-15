# Data handling
import pandas as pd
import json
import itertools
import time
import re

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Warning control
import warnings
warnings.filterwarnings("ignore")


def make_safe_filename(s):
    s = s.replace("→", "to")
    s = s.replace("|", "_")
    s = re.sub(r"\s+", "_", s)  
    s = re.sub(r"[^a-zA-Z0-9_\(\)\-]", "", s)  
    s = re.sub(r"_+", "_", s)  
    return s.strip("_")

def get_best_sarima(ts, p, d, q, P, D, Q, s):
    #Performs grid search over SARIMA parameters to find the best configuration (based on AIC).
    
    best_aic = float("inf")
    best_order = None
    best_seasonal = None

    # Iterate through all parameter combinations
    for order in itertools.product(p, d, q):
        for seasonal in itertools.product(P, D, Q):
            try:
                model = SARIMAX(ts, order=order, seasonal_order=seasonal + (s,))
                fit = model.fit(disp=False)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = order
                    best_seasonal = seasonal
            except:
                continue  # skip invalid models

    # Return the best combination found
    if best_order and best_seasonal:
        return {
            "order": list(best_order),
            "seasonal_order": list(best_seasonal) + [s]
        }
    return None


def create_sarima_param_file(df, routes, output_file="Generated/custom_sarima_params.json"):

    #Generates and saves optimal SARIMA parameters for each route–airline combination, and also for route-level aggregation (across airlines).
    config = {}

    # Prepare DATE and ROUTE columns
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    df = df.sort_values("DATE")

    # Define SARIMA parameter grid
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = 12  # monthly seasonality

    # Get unique route–airline triples to evaluate separately
    route_airline_aircraft_triples = (
        df[df["ROUTE"].isin(routes)][["ROUTE", "UNIQUE_CARRIER_NAME", "AIRCRAFT_TYPE"]]
        .dropna()
        .drop_duplicates()
        .values
        .tolist()
    )

    print(f"Generating SARIMA parameters for {len(route_airline_aircraft_triples)} route–airline-aircraft-triples...")
    start = time.time()

    for idx, (route, airline, aircraft) in enumerate(route_airline_aircraft_triples):
        sub_df = df[(df["ROUTE"] == route) & (df["UNIQUE_CARRIER_NAME"] == airline) & (df["AIRCRAFT_TYPE"] == aircraft)]
        if sub_df.shape[0] < 18:
            continue  # not enough data points

        # Create a clean monthly time series
        ts = (
            sub_df
            .groupby("DATE", as_index=True)["PASSENGERS"]
            .sum()
            .asfreq("MS")
        )

        key = f"{route} | {airline} | {aircraft}"  # use route + airline as key
        safe_key = make_safe_filename(key)
        best = get_best_sarima(ts, p, d, q, P, D, Q, s)
        if best:
            config[safe_key] = best

        if idx % 10 == 0:
            print(f"  ⏳ {idx}/{len(route_airline_aircraft_triples)} processed – {time.time() - start:.1f}s elapsed")

    print("Generating aggregated SARIMA parameters per route (across airlines)...")

    # Also create SARIMA models for aggregated route-level series (all airlines combined)
    for route in routes:
        sub_df = df[df["ROUTE"] == route]
        if sub_df.shape[0] < 18:
            continue

        ts = (
            sub_df
            .groupby("DATE", as_index=True)["PASSENGERS"]
            .sum()
            .asfreq("MS")
        )

        key = route  # route-only key for fallback
        best = get_best_sarima(ts, p, d, q, P, D, Q, s)
        if best:
            config[key] = best

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"SARIMA config saved to: {output_file} ({len(config)} entries)")


if __name__ == "__main__":
    # Load preprocessed airline data
    df = pd.read_csv("Generated/Grouped_All_Valid_Connections.csv", low_memory=False)

    # Create ROUTE and DATE columns
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]

    # Select top N routes by total passenger volume
    top_routes = (
        df.groupby("ROUTE")["PASSENGERS"]
        .sum()
        .sort_values(ascending=False)
        .head(10)  # adjust as needed
        .index
        .tolist()
    )

    # Run parameter generation
    create_sarima_param_file(df, top_routes)
