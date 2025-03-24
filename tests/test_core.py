import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import faiss

from app.chatbot import find_similar_stations, generate_llm_response
from etl.fetch_data import fetch_page_data
from utilities.utils import get_latest_processed_file


# -----------------------
# 1. Test FAISS similarity
# -----------------------
def test_find_similar_stations_returns_expected_columns():
    """Ensure that FAISS search returns DataFrame with expected structure."""
    # Mock FAISS index with 2D dummy vectors
    dim = 3
    index = faiss.IndexFlatL2(dim)
    vectors = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype='float32')
    index.add(vectors)

    # Dummy metadata DataFrame aligned with index
    metadata_df = pd.DataFrame({
        "station_id": [1, 2],
        "text": ["Station A", "Station B"]
    })

    # Dummy embed_model that returns the same dimension
    class DummyEmbedModel:
        def encode(self, text, convert_to_numpy=True):
            return np.array([[0.15, 0.25, 0.35]])

    results = find_similar_stations("Some query", index, metadata_df, DummyEmbedModel(), k=1)
    
    assert "station_id" in results.columns
    assert "distance" in results.columns
    assert len(results) == 1


# -----------------------
# 2. Test LLM generation
# -----------------------
def test_generate_llm_response_structure():
    """Validate that LLM response includes expected keywords."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()

    prompt = "Answer: What is 2 + 2?"
    response = generate_llm_response(prompt, tokenizer, model, max_new_tokens=10)

    assert isinstance(response, str)
    assert len(response) > 0


# -----------------------
# 3. Test fetch_data API call
# -----------------------
@patch("etl.fetch_data.requests.get")
def test_fetch_page_data_success(mock_get):
    """Check successful fetch from mocked API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": 1}]}
    mock_get.return_value = mock_response

    from etl.fetch_data import fetch_page_data
    result = fetch_page_data(1)

    assert "data" in result
    assert isinstance(result["data"], list)


@patch("etl.fetch_data.requests.get")
def test_fetch_page_data_failure(mock_get):
    """Ensure error handling for failed fetch."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    result = fetch_page_data(1)
    assert result is None


# -----------------------
# 4. Test utility function
# -----------------------
def test_get_latest_processed_file(tmp_path):
    """Validate latest file selection from dummy data."""
    # Create fake data files
    file1 = tmp_path / "processed_data_2025-03-20.csv"
    file2 = tmp_path / "processed_data_2025-03-23.csv"
    file1.write_text("dummy")
    file2.write_text("dummy")

    path, date = get_latest_processed_file(str(tmp_path))
    assert "2025-03-23" in path
    assert date.strftime("%Y-%m-%d") == "2025-03-23"
