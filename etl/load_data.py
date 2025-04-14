import os
import sqlite3
import pandas as pd
import logging
from utilities.utils import get_latest_processed_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_database(data_path):
    """
    Create the SQLite database and ensure the fuel_prices table exists.
    """
    DB_PATH = os.path.join(data_path, "fuel_prices.db")
    TABLE_NAME = "fuel_prices"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER,
            latitude REAL,
            longitude REAL,
            postal_code TEXT,
            address TEXT,
            city TEXT,
            department TEXT,
            code_department TEXT,
            region TEXT,
            services TEXT,
            gazole_price REAL,
            gazole_maj DATE,
            e10_price REAL,
            e10_maj DATE,
            sp98_price REAL,
            sp98_maj DATE,
            sp95_price REAL,
            sp95_maj DATE,
            e85_price REAL,
            e85_maj DATE,
            gplc_price REAL,
            gplc_maj DATE,
            gazole_unavailable TEXT,
            e10_unavailable TEXT,
            sp98_unavailable TEXT,
            sp95_unavailable TEXT,
            e85_unavailable TEXT,
            gplc_unavailable TEXT,
            updated_at DATE
        );
    """)
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")

def get_latest_database_updates(db_path, table_name):
    """
    Retrieve the latest record per station_id from the database.
    """
    conn = sqlite3.connect(db_path)
    query = f"""
        WITH LatestUpdates AS (
            SELECT *, MAX(updated_at) OVER (PARTITION BY station_id) AS max_updated_at
            FROM {table_name}
        )
        SELECT * FROM LatestUpdates WHERE updated_at = max_updated_at;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date columns to datetime
    date_columns = ["gazole_maj", "e10_maj", "sp98_maj", "sp95_maj", "e85_maj", "gplc_maj"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def identify_updated_station_ids(latest_fetched_data_df, latest_db_data, date_columns):
    """
    Identify stations with updated fuel price timestamps.
    """
    updated_station_ids = []
    for station_id in latest_fetched_data_df['station_id']:
        latest_filtered = latest_fetched_data_df[latest_fetched_data_df["station_id"] == station_id]
        db_filtered = latest_db_data[latest_db_data["station_id"] == station_id]
        
        if not latest_filtered.empty and not db_filtered.empty:
            latest_values = latest_filtered.iloc[0][date_columns].apply(pd.to_datetime, errors='coerce')
            db_values = db_filtered.iloc[0][date_columns].apply(pd.to_datetime, errors='coerce')
            
            if any(pd.notna(latest_values[col]) and pd.notna(db_values[col]) and latest_values[col] > db_values[col] for col in date_columns):
                updated_station_ids.append(station_id)
    
    logging.info(f'Total updated stations: {len(updated_station_ids)}')
    return updated_station_ids

def load_latest_processed_data(data_path):
    """
    Load the latest processed fuel price data and update the database with new or modified records.
    """
    DB_PATH = os.path.join(data_path, "fuel_prices.db")
    TABLE_NAME = "fuel_prices"
    DATE_COLUMNS = ["gazole_maj", "e10_maj", "sp98_maj", "sp95_maj", "e85_maj", "gplc_maj"]

    latest_db_data = get_latest_database_updates(DB_PATH, TABLE_NAME)
    filename_starts_with = "processed_data"
    latest_processed_file, latest_processed_date = get_latest_processed_file(data_path, filename_starts_with)

    if not latest_processed_file or not os.path.exists(latest_processed_file):
        logging.warning("No processed data file found. Ensure transform_data.py has been run.")
        return

    latest_fetched_data_df = pd.read_csv(latest_processed_file, parse_dates=DATE_COLUMNS)
    new_station_ids = [i for i in latest_fetched_data_df['station_id'] if i not in latest_db_data['station_id'].values]
    latest_fetched_data_df_without_new = latest_fetched_data_df[~latest_fetched_data_df['station_id'].isin(new_station_ids)]
    updated_station_ids = identify_updated_station_ids(latest_fetched_data_df_without_new, latest_db_data, DATE_COLUMNS)
    
    new_or_updated_station_ids = new_station_ids + updated_station_ids
    logging.info(f'Total new or updated stations to add: {len(new_or_updated_station_ids)}')
    
    df_new = latest_fetched_data_df[latest_fetched_data_df['station_id'].isin(new_or_updated_station_ids)]
    
    if df_new.empty:
        logging.info("No new updates found. Database is already up to date.")
    else:
        conn = sqlite3.connect(DB_PATH)
        df_new.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.close()
        logging.info(f"Inserted {len(df_new)} new or updated records into the database.")

def run_load_data(data_path):
    """
    Run the load data process.
    """ 
    initialize_database(data_path)
    load_latest_processed_data(data_path)
