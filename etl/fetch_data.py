import requests
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API Constants
BASE_URL = "https://tabular-api.data.gouv.fr/api/resources/edd67f5b-46d0-4663-9de9-e5db1c880160/data/"
PAGE_SIZE = 50  # Number of records per API call

# Generate file name with current date (without hour)
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"data/raw_data_{CURRENT_DATE}.json"  # Dynamic filename

def fetch_page_data(page: int) -> dict:
    """
    Fetch fuel price data from API for a given page.

    Args:
        page (int): Page number to fetch.
    
    Returns:
        dict: JSON response data if successful, otherwise None.
    """
    url = f"{BASE_URL}?page={page}&page_size={PAGE_SIZE}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error {response.status_code}: Unable to fetch data from page {page}.")
        return None

def fetch_all_data() -> list:
    """
    Fetch all fuel price data using pagination.

    Returns:
        list: List of all fuel price records fetched from the API.
    """
    page = 1
    all_data = []

    while True:
        logging.info(f"Fetching page {page}...")
        data = fetch_page_data(page)

        if not data or 'data' not in data or not data['data']:
            logging.info("No more data to fetch. Stopping.")
            break

        all_data.extend(data['data'])

        if not data.get("links", {}).get("next"):
            logging.info("Reached last page. Stopping extraction.")
            break

        page += 1

    logging.info(f"Total records fetched: {len(all_data)}")
    return all_data

def save_data_to_file(data: list, output_file: str):
    """
    Save extracted data to a JSON file.

    Args:
        data (list): List of fuel price records.
        output_file (str): Path to the output JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    logging.info(f"Data extraction complete: {len(data)} records saved to {output_file}.")

def extract_data():
    """
    Main function to fetch and save fuel price data.
    """
    data = fetch_all_data()
    if data:
        save_data_to_file(data, OUTPUT_FILE)
    else:
        logging.warning("No data extracted.")

if __name__ == "__main__":
    extract_data()
