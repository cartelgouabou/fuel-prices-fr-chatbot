import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fuel_data_processing.log"),
        logging.StreamHandler()
    ]
)

DB_PATH = "data/fuel_prices.db"
VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"

# Load pre-trained embedding model
logging.info("Loading sentence transformer model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logging.info("Model loaded successfully.")

def get_fuel_data():
    """Retrieve the latest fuel price data from SQLite"""
    logging.info("Connecting to the database...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            WITH LatestFuelData AS (
                SELECT station_id, address, city, code_department, region, gazole_price, sp98_price, sp95_price, gplc_price, e10_price, e85_price, updated_at,
                MAX(updated_at) OVER (PARTITION BY station_id) AS max_updated_at
                FROM fuel_prices
            )
            SELECT station_id, address, city, code_department, region, gazole_price, sp98_price, sp95_price, gplc_price, e10_price, e85_price, updated_at
            FROM LatestFuelData
            WHERE updated_at = max_updated_at;
        """
        df = pd.read_sql_query(query, conn)
        logging.info(f"Retrieved {len(df)} latest records from the database.")
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
        logging.info("Database connection closed.")
    return df

def generate_embeddings():
    """Generate embeddings for the latest fuel price data"""
    logging.info("Generating embeddings for fuel price data...")
    df = get_fuel_data()
    if df.empty:
        logging.warning("No data retrieved. Exiting process.")
        return

    df["text"] = df.apply(lambda row: f"Station {row.station_id} in {row.address}, {row.city}, {row.code_department} , {row.region}: \
        Gazole={row.gazole_price}, SP98={row.sp98_price}, SP95={row.sp95_price}, E10={row.e10_price}, E85={row.e85_price}, GPLC={row.gplc_price}.", axis=1)
    
    logging.info("Encoding data using sentence transformer...")
    embeddings = embed_model.encode(df["text"].tolist(), convert_to_numpy=True)
    logging.info("Embeddings generated successfully.")
    
    # Save FAISS index
    logging.info("Creating and saving FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)
    logging.info(f"FAISS index saved to {VECTOR_STORE_PATH}.")
    
    # Save metadata
    logging.info("Saving embeddings metadata...")
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(df, f)
    logging.info(f"Metadata saved to {EMBEDDINGS_FILE}.")
    
    logging.info(f"Updated embeddings and FAISS index with {len(df)} fuel records.")

if __name__ == "__main__":
    logging.info("Starting embedding generation process...")
    generate_embeddings()
    logging.info("Process completed.")
