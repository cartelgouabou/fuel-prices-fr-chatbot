import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from etl.fetch_data import fetch_data, extract_data

@pytest.mark.parametrize("mock_response,expected_count", [
    ({"data": [{"id": 101, "fuel_type": "SP98", "price": 1.89}], "links": {"next": None}}, 1),
    ({"data": [], "links": {"next": None}}, 0)
])
@patch("requests.get")
def test_fetch_data(mock_get, mock_response, expected_count):
    """Test that fetch_data correctly retrieves data and handles empty responses"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    result = fetch_data(page=1)
    assert result is not None
    assert len(result["data"]) == expected_count


@patch("requests.get")
def test_fetch_data_api_error(mock_get):
    """Test handling of API error responses"""
    mock_get.return_value.status_code = 500  # Simulating API failure

    result = fetch_data(page=1)
    assert result is None  # Should return None on error


@patch("requests.get")
def test_extract_data_pagination(mock_get):
    """Test that extract_data correctly handles paginated API responses"""
    mock_response_page_1 = {
        "data": [{"id": 101, "fuel_type": "SP98", "price": 1.89}],
        "links": {"next": "fake_url"}
    }
    mock_response_page_2 = {
        "data": [{"id": 102, "fuel_type": "Gazole", "price": 1.75}],
        "links": {"next": None}  # No further pages
    }

    # Simulate two API calls: first returns page 1, second returns page 2
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: mock_response_page_1),
        MagicMock(status_code=200, json=lambda: mock_response_page_2)
    ]

    result = extract_data()
    assert len(result) == 2  # Two records from two pages
    assert result[0]["id"] == 101
    assert result[1]["id"] == 102


@patch("requests.get")
def test_extract_data_end_condition(mock_get):
    """Test that extract_data stops fetching when 'next' is null"""
    mock_response = {
        "data": [{"id": 101, "fuel_type": "SP98", "price": 1.89}],
        "links": {"next": None}  # End condition reached
    }
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    result = extract_data()

    # Ensure function returns the expected number of records
    assert len(result) == 1
    assert result[0]["id"] == 101

    # Ensure API was called only once (pagination stopped)
    assert mock_get.call_count == 1, "API should be called only once if 'next' is null"
