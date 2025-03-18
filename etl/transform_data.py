import json
import os
import pandas as pd
import logging
from datetime import datetime
from utilities.utils import get_latest_processed_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_latest_raw_data(data_path, filename_starts_with):
    """
    Retrieve the latest raw JSON data file and load its content.

    Args:
        data_path (str): Directory where raw data files are stored.
        filename_starts_with (str): Prefix of the raw data files.

    Returns:
        tuple: List of raw records and the fetch date.
    """
    latest_raw_file, latest_fetch_date = get_latest_processed_file(data_path, filename_starts_with)

    if not latest_raw_file or not latest_fetch_date:
        logging.warning("No valid raw data file found. Run fetch_data.py first.")
        return None, None

    with open(latest_raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    return raw_data, latest_fetch_date

def extract_fuel_data(record, latest_fetch_date):
    """
    Extract relevant fuel station details from a raw record.

    Args:
        record (dict): A single fuel station record from raw data.
        latest_fetch_date (str): The date when the raw data was fetched.

    Returns:
        dict: Processed fuel station data.
    """
    station_id = record.get("id")
    latitude = record.get("latitude", 0) / 100000  # Normalize lat/lon
    longitude = record.get("longitude", 0) / 100000  # Normalize lat/lon
    postal_code = record.get("Code postal")
    address = record.get("Adresse")
    city = record.get("Ville")
    department = record.get("Département")
    code_department = record.get("code_departement")
    region = record.get("Région")
    
    # Extract services
    services = None
    if isinstance(record.get("services"), dict):
        services = ", ".join(record["services"].get("service", []))
    
    # Extract fuel prices and update timestamps
    fuel_prices, fuel_majs = {}, {}
    if isinstance(record.get("prix"), list):
        fuel_prices = {fuel["@nom"]: fuel.get("@valeur") for fuel in record["prix"]}
        fuel_majs = {fuel["@nom"]: fuel.get("@maj") for fuel in record["prix"]}
    
    # Extract fuel unavailability status
    fuel_unavailable = {}
    if isinstance(record.get("rupture"), list):
        fuel_unavailable = {fuel["@nom"]: fuel.get("@type") for fuel in record["rupture"]}
    
    return {
        "station_id": station_id,
        "latitude": latitude,
        "longitude": longitude,
        "postal_code": postal_code,
        "address": address,
        "city": city,
        "department": department,
        "code_department": code_department,
        "region": region,
        "services": services,
        "gazole_price": float(fuel_prices.get("Gazole", 0)) if fuel_prices.get("Gazole") else None,
        "gazole_maj": fuel_majs.get("Gazole"),
        "e10_price": float(fuel_prices.get("E10", 0)) if fuel_prices.get("E10") else None,
        "e10_maj": fuel_majs.get("E10"),
        "sp98_price": float(fuel_prices.get("SP98", 0)) if fuel_prices.get("SP98") else None,
        "sp98_maj": fuel_majs.get("SP98"),
        "sp95_price": float(fuel_prices.get("SP95", 0)) if fuel_prices.get("SP95") else None,
        "sp95_maj": fuel_majs.get("SP95"),
        "e85_price": float(fuel_prices.get("E85", 0)) if fuel_prices.get("E85") else None,
        "e85_maj": fuel_majs.get("E85"),
        "gplc_price": float(fuel_prices.get("GPLc", 0)) if fuel_prices.get("GPLc") else None,
        "gplc_maj": fuel_majs.get("GPLc"),
        "gazole_unavailable": fuel_unavailable.get("Gazole"),
        "e10_unavailable": fuel_unavailable.get("E10"),
        "sp98_unavailable": fuel_unavailable.get("SP98"),
        "sp95_unavailable": fuel_unavailable.get("SP95"),
        "e85_unavailable": fuel_unavailable.get("E85"),
        "gplc_unavailable": fuel_unavailable.get("GPLc"),
        "updated_at": latest_fetch_date
    }

def transform_data(data_path):
    """
    Load the latest raw JSON, clean data, and save processed data to CSV.
    """
    raw_data, latest_fetch_date = load_latest_raw_data(data_path, "raw_data")
    if not raw_data:
        return None
    
    processed_records = [extract_fuel_data(record, latest_fetch_date) for record in raw_data]
    df = pd.DataFrame(processed_records)
    
    # Remove duplicate and incomplete records
    df = df.dropna(subset=["station_id", "latitude", "longitude"])
    
    # Save cleaned data to CSV
    processed_data_file = os.path.join(data_path, f"processed_data_{latest_fetch_date}.csv")
    os.makedirs(data_path, exist_ok=True)
    df.to_csv(processed_data_file, index=False, encoding="utf-8")
    
    logging.info(f"Data transformed and saved: {len(df)} records in {processed_data_file}")
    return df

if __name__ == "__main__":
    data_path = r"E:\My_Github\fr-fuel-price-tracking\data"
    transform_data(data_path)