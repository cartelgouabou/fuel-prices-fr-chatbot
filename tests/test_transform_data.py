import sys
import os
import pytest
import json
import pandas as pd

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.transform_data import transform_data

# Define temporary test files to avoid modifying real data
TEST_RAW_DATA_FILE = "tests/temp_raw_data.json"
TEST_PROCESSED_DATA_FILE = "tests/temp_processed_data.csv"


@pytest.fixture
def setup_test_files():
    """Fixture to create temporary raw data file before test and clean up after."""
    raw_data_sample = [
        {
            "id": 101, "latitude": 483217, "longitude": 227631,
            "Code postal": "75001", "Adresse": "10 Rue Test", "Ville": "Paris",
            "D\u00e9partement": "Paris", "code_departement": "75",
            "R\u00e9gion": "Île-de-France",
            "services": {"service": ["Carburants", "Boutique"]},
            "prix": [{"@nom": "SP98", "@valeur": "1.89", "@maj": "2024-06-10"}],
            "rupture": [{"@nom": "Gazole", "@type": "rupture"}]
        }
    ]
    os.makedirs(os.path.dirname(TEST_RAW_DATA_FILE), exist_ok=True)
    with open(TEST_RAW_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_data_sample, f)

    yield  # Run the tests

    # Cleanup after tests
    if os.path.exists(TEST_RAW_DATA_FILE):
        os.remove(TEST_RAW_DATA_FILE)
    if os.path.exists(TEST_PROCESSED_DATA_FILE):
        os.remove(TEST_PROCESSED_DATA_FILE)


def test_transform_data_output_format(setup_test_files):
    """Test that transform_data outputs a valid Pandas DataFrame"""
    df = transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # Ensure at least one record is processed


def test_transform_data_lat_lon_scaling(setup_test_files):
    """Test latitude and longitude normalization (should be in decimal degrees)"""
    df = transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert round(df.iloc[0]["latitude"], 5) == 4.83217
    assert round(df.iloc[0]["longitude"], 5) == 2.27631


def test_transform_data_fuel_price_extraction(setup_test_files):
    """Test that fuel prices are correctly extracted and converted to float"""
    df = transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert isinstance(df.iloc[0]["sp98_price"], float)
    assert df.iloc[0]["sp98_price"] == 1.89  # Ensure price conversion works


@pytest.mark.parametrize("sample_data,expected_price", [
    ([{"id": 102, "latitude": 500000, "longitude": 200000, "prix": []}], None),
    ([{"id": 103, "latitude": 500000, "longitude": 200000, "prix": [{"@nom": "SP98", "@valeur": "2.05"}]}], 2.05)
])
def test_transform_data_handle_missing_prices(sample_data, expected_price):
    """Test handling of missing or available fuel prices"""
    with open(TEST_RAW_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    df = transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert "sp98_price" in df.columns
    assert df.iloc[0]["sp98_price"] == expected_price


def test_transform_data_file_saving(setup_test_files):
    """Test that transformed data is correctly saved to CSV"""
    transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert os.path.exists(TEST_PROCESSED_DATA_FILE)

    # Read the saved CSV and check contents
    df_saved = pd.read_csv(TEST_PROCESSED_DATA_FILE)
    assert len(df_saved) > 0  # Ensure data was written to CSV


def test_transform_data_handle_missing_file():
    """Test that function handles missing raw data file gracefully"""
    if os.path.exists(TEST_RAW_DATA_FILE):
        os.remove(TEST_RAW_DATA_FILE)  # Ensure file is missing

    df = transform_data(TEST_RAW_DATA_FILE, TEST_PROCESSED_DATA_FILE)
    assert df is None  # Should return None if file is missing
