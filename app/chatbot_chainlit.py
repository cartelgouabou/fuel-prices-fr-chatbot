import os
import logging
import pickle
import faiss
import html
import json
import chainlit as cl
import requests
from datetime import datetime
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
import pandas as pd

# ========== Setup ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"

MODEL_MAP = {
    "mistral": "Ollama (Mistral)",
    "gemma": "Ollama (Gemma)"
}

# ========== Global Variables ==========
embed_model = None
faiss_index = None
metadata_df = None
city_matcher = None
dept_matcher = None
region_matcher = None

# ========== Initialization ==========
def is_ollama_running() -> bool:
    try:
        #requests.get("http://localhost:11434", timeout=2).raise_for_status()
        requests.get("http://host.docker.internal:11434", timeout=2).raise_for_status() # For Docker
        return True
    except requests.RequestException:
        return False

def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_faiss_index_and_metadata():
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(EMBEDDINGS_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def build_location_matchers(metadata_df):
    def init_matcher(column):
        matcher = KeywordProcessor()
        for item in metadata_df[column].dropna().unique():
            matcher.add_keyword(str(item).lower())
        return matcher

    return (
        init_matcher("city"),
        init_matcher("code_department"),
        init_matcher("region")
    )

def detect_location(query, city_matcher, dept_matcher, region_matcher):
    q = query.lower()
    return {
        "city": city_matcher.extract_keywords(q)[0] if city_matcher.extract_keywords(q) else None,
        "department": dept_matcher.extract_keywords(q)[0] if dept_matcher.extract_keywords(q) else None,
        "region": region_matcher.extract_keywords(q)[0] if region_matcher.extract_keywords(q) else None,
    }

def search_with_filter(query, metadata_df, faiss_index, matchers):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vec, k=30)
    results = metadata_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    city_matcher, dept_matcher, region_matcher = matchers
    detected = detect_location(query, city_matcher, dept_matcher, region_matcher)

    filtered = results
    if detected["city"]:
        filtered = results[results["city"].str.lower() == detected["city"]]
    elif detected["department"]:
        filtered = results[results["code_department"].astype(str) == detected["department"]]
    elif detected["region"]:
        filtered = results[results["region"].str.lower() == detected["region"]]

    fallback = filtered.empty
    return (filtered.head(15) if not fallback else results.head(5)), fallback

def get_latest_fetch_date():
    try:
        files = [f for f in os.listdir("data") if f.startswith("raw_data_")]
        dates = [
            datetime.strptime(f.split("_")[-1].replace(".json", ""), "%Y-%m-%d").date()
            for f in files
        ]
        return max(dates) if dates else None
    except Exception as e:
        logging.warning(f"Failed to extract fetch date: {e}")
        return None

def generate_llm_response(prompt, model_name="mistral", max_new_tokens=200):
    try:
        response = requests.post(
            #"http://localhost:11434/api/generate",
            "http://host.docker.internal:11434/api/generate", # For Docker
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {"temperature": 0.7, "num_predict": max_new_tokens},
            },
            stream=True,
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                full_response += data.get("response", "")
        return full_response.strip()
    except Exception as e:
        logging.error(f"Ollama generation error: {e}")
        return f"❌ Error generating response: {str(e)}"

# ========== Chainlit Handlers ==========

@cl.on_chat_start
async def init():
    global embed_model, faiss_index, metadata_df, city_matcher, dept_matcher, region_matcher

    await cl.Message(content="Initializing chatbot...").send()

    embed_model = load_embed_model()
    faiss_index, metadata_df = load_faiss_index_and_metadata()
    city_matcher, dept_matcher, region_matcher = build_location_matchers(metadata_df)

    await cl.Message(content="✅ Ready! Ask me about fuel prices in France (e.g., 'Cheapest SP98 in Marseille')").send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    model = "mistral"  # You can use UI elements to select this later

    if not is_ollama_running():
        await cl.Message(content="❌ Ollama server not running. Please start it using `ollama run mistral` or `ollama run gemma`.").send()
        return

    await cl.Message(content="🔍 Searching and preparing response...").send()

    results, fallback = search_with_filter(query, metadata_df, faiss_index, (city_matcher, dept_matcher, region_matcher))
    latest_fetch_date = get_latest_fetch_date()
    context = "\n".join(results["text"].tolist())

    prompt = (
        f"Here is a list of French fuel station info fetched on {latest_fetch_date}:\n"
        f"{context}\n\nUser question: {query}\nProvide a clear and concise answer."
    )
    response = generate_llm_response(prompt, model_name=model)

    header = "📌 No exact location match found — showing top semantic results." if fallback else "✅ Results based on matched location."
    await cl.Message(content=f"### 🤖 Chatbot Response\n{response}").send()
    await cl.Message(content=f"### 📄 Context Given to LLM\nFetch Date: {latest_fetch_date}\n```{context}```").send()
