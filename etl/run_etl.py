import os
import logging
from etl.fetch_data import extract_data
from etl.transform_data import transform_data
from etl.load_data import run_load_data
from etl.process_fuel_embeddings import generate_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_etl():
    """
    Execute the full ETL pipeline: Extract, Transform, and Load.
    """
    logging.info("Starting ETL process...")
    # Automatically define data_path as '../data'
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    
    try:
        steps = [
            ("Extracting data", extract_data),
            ("Transforming data", lambda: transform_data(data_path)),
            ("Initializing database", lambda: run_load_data(data_path)),
            ("Update the FAISS embeddings for chatbot", generate_embeddings)
        ]
        
        for step_name, step_function in steps:
            logging.info(f"{step_name}...")
            step_function()
            logging.info(f"{step_name} completed successfully.")
        
        logging.info("ETL process completed successfully.")
    
    except Exception as e:
        logging.error(f"ETL process failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_etl()
