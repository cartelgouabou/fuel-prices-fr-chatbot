import logging
from etl.fetch_data import extract_data
from etl.transform_data import transform_data
from etl.load_data import load_latest_processed_data, initialize_database

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_etl():
    """
    Execute the full ETL pipeline: Extract, Transform, and Load.
    """
    logging.info("Starting ETL process...")
    data_path = r"E:\My_Github\fr-fuel-price-tracking\data"
    
    try:
        steps = [
            ("Extracting data", extract_data),
            ("Transforming data", lambda: transform_data(data_path)),
            ("Initializing database", lambda: initialize_database(data_path)),
            ("Loading data into database", lambda: load_latest_processed_data(data_path))
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
