import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import time

# Configuration
DB_PATH = "data/fuel_prices.db"
VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fuel_data_processing.log"),
        logging.StreamHandler()
    ]
)

def get_fuel_data():
    """Retrieve the latest fuel price data from SQLite."""
    logging.info("Connecting to the database...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                WITH LatestFuelData AS (
                    SELECT station_id, address, city, code_department, region,
                           gazole_price, sp98_price, sp95_price, gplc_price, e10_price, e85_price, updated_at,
                           MAX(updated_at) OVER (PARTITION BY station_id) AS max_updated_at
                    FROM fuel_prices
                )
                SELECT station_id, address, city, code_department, region,
                       gazole_price, sp98_price, sp95_price, gplc_price, e10_price, e85_price, updated_at
                FROM LatestFuelData
                WHERE updated_at = max_updated_at;
            """
            df = pd.read_sql_query(query, conn)
            logging.info(f"Retrieved {len(df)} latest records from the database.")
            return df
    except Exception as e:
        logging.error(f"Error retrieving data: {e}", exc_info=True)
        return pd.DataFrame()

def generate_embeddings():
    """Generate FAISS index and save fuel station metadata with sentence embeddings."""
    start_time = time.time()
    logging.info("Starting embedding generation process...")

    df = get_fuel_data()
    if df.empty:
        logging.warning("No data available to generate embeddings.")
        return

    df["text"] = df.apply(lambda row: (
        f"📍 {row.city.upper() if row.city else ''} station id {row.station_id} located at "
        f"{row.address.lower() if row.address else ''}, {row.code_department}, {row.region}./ "
        f"Prices: Gazole={row.gazole_price}, SP98={row.sp98_price}, SP95={row.sp95_price}, "
        f"E10={row.e10_price}, E85={row.e85_price}, GPLC={row.gplc_price}."
    ), axis=1)

    # df["text"] = df.apply(lambda row: (
    #     f"📍 Located in {row.city.upper() if row.city else ''} department {row.code_department}, region {row.region}. "
    #     f"Prices: Gazole={row.gazole_price}, SP98={row.sp98_price}, SP95={row.sp95_price}, "
    #     f"E10={row.e10_price}, E85={row.e85_price}, GPLC={row.gplc_price}."
    # ), axis=1)

    logging.info("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    logging.info("Generating embeddings...")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

    logging.info("Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(df, f)

    elapsed = time.time() - start_time
    logging.info(f"Embeddings generated for {len(df)} records and saved in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    generate_embeddings()
