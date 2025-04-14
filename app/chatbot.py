import os
import logging
import pickle
import faiss
import html
import json
import streamlit as st
import requests
from datetime import datetime
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
import ollama
import pandas as pd

# ========== Setup ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"


def is_ollama_running() -> bool:
    try:
        #requests.get("http://localhost:11434", timeout=2).raise_for_status() # Uncomment on manual setup
        requests.get("http://host.docker.internal:11434", timeout=2).raise_for_status() # Uncomment on Docker compose setup
        return True
    except requests.RequestException:
        return False

if "ollama_initialized" not in st.session_state:
    if is_ollama_running():
        st.session_state.ollama_initialized = True
    else:
        logging.warning("\u26a0\ufe0f Ollama not running. Skipping model pull.")

# ========== Cached Resources ==========
@st.cache_resource(show_spinner=False)
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_faiss_index_and_metadata() -> tuple[faiss.Index, pd.DataFrame]:
    try:
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(EMBEDDINGS_FILE, "rb") as f:
            metadata_df = pickle.load(f)
        return index, metadata_df
    except Exception as e:
        st.error(f"❌ Error loading FAISS index or embeddings: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def build_location_matchers(metadata_df: pd.DataFrame) -> tuple[KeywordProcessor, KeywordProcessor, KeywordProcessor]:
    def init_matcher(column: str) -> KeywordProcessor:
        matcher = KeywordProcessor()
        for item in metadata_df[column].dropna().unique():
            matcher.add_keyword(str(item).lower())
        return matcher

    return (
        init_matcher("city"),
        init_matcher("code_department"),
        init_matcher("region"),
    )

# ========== Helper Functions ==========
def get_latest_fetch_date() -> datetime | None:
    try:
        data_path = os.path.join(os.path.dirname(__file__), "../data")
        files = [f for f in os.listdir(data_path) if f.startswith("raw_data_")]
        dates = [
            datetime.strptime(f.split("_")[-1].replace(".json", ""), "%Y-%m-%d").date()
            for f in files
        ]
        return max(dates) if dates else None
    except Exception as e:
        logging.warning(f"Failed to extract fetch date: {e}")
        return None

def detect_location(query: str, city_matcher, dept_matcher, region_matcher) -> dict:
    q = query.lower()
    return {
        "city": city_matcher.extract_keywords(q)[0] if city_matcher.extract_keywords(q) else None,
        "department": dept_matcher.extract_keywords(q)[0] if dept_matcher.extract_keywords(q) else None,
        "region": region_matcher.extract_keywords(q)[0] if region_matcher.extract_keywords(q) else None,
    }

def search_with_filter(query: str, embed_model, metadata_df, faiss_index, matchers) -> tuple[pd.DataFrame, bool]:
    city_matcher, dept_matcher, region_matcher = matchers
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vec, k=30)

    results = metadata_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
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

def generate_llm_response(prompt: str, backend="Ollama (Mistral)", max_new_tokens=200) -> str:
    model_map = {
        "Ollama (Mistral)": "mistral",
        "Ollama (Gemma)": "gemma",
    }
    model = model_map.get(backend, "mistral")

    try:
        response = requests.post(
            #"http://localhost:11434/api/generate", # Uncomment on manual setup
            "http://host.docker.internal:11434/api/generate", # Uncomment on Docker compose setup
            json={
                "model": model,
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
        return f"❌ Error generating response from {model}: {str(e)}"

def enhance_query_with_llm(user_query: str, backend="Ollama (Mistral)", max_new_tokens=100) -> str:
    enhancement_prompt = f"""
You are optimizing a search query to retrieve fuel station data in France. The data is structured like this:

📍 CITY station id 12345 located at ADDRESS, DEPT_CODE, REGION./ 
Prices: Gazole=..., SP98=..., SP95=..., E10=..., E85=..., GPLC=...

Based on the following user query: **\"{user_query}\"**

- Detect and insert any known or implied fuel types
- Clarify vague terms like \"cheapest\" or \"best\" with measurable intent (e.g., \"lowest price\")
- Insert any detected or inferred location like city, department, or region
- Output only the query in a structured way to match the context format above

Enhanced query:"""

    return generate_llm_response(enhancement_prompt, backend=backend, max_new_tokens=max_new_tokens)

# ========== Streamlit UI ==========
st.set_page_config(page_title="⛽ Fuel Prices Chatbot", layout="wide")

embed_model = load_embed_model()
index, metadata_df = load_faiss_index_and_metadata()
location_matchers = build_location_matchers(metadata_df)

# --- Header ---
st.markdown("""
    <h1 style='text-align: center;'>⛽ France Fuel Prices Chatbot</h1>
    <p style='text-align: center; font-size: 18px;'>Ask me anything about fuel prices in France. I'll search and summarize intelligently!</p>
""", unsafe_allow_html=True)
st.markdown("---")

# --- User Input ---
col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox("🧠 Choose LLM Backend", ["Ollama (Mistral)", "Ollama (Gemma)"], index=0)
    user_query = st.text_input("💬 What do you want to know?", placeholder="e.g. Cheapest SP98 in Marseille?")
    search_clicked = st.button("🔍 Search")
with col2:
    st.image("https://img.icons8.com/color/96/gas-pump.png", width=96)
    st.caption(f"Powered by MiniLM + FAISS + {model_choice}")

# --- Main Logic ---
if search_clicked and user_query:
    if not is_ollama_running():
        st.error(f"❌ Ollama server is not running.\nPlease start it with: `ollama run {model_choice.split()[-1].lower()}`")
        st.stop()

    with st.spinner("🔎 Searching and generating response..."):
        enhanced_query = enhance_query_with_llm(user_query, backend=model_choice)
        results, fallback = search_with_filter(enhanced_query, embed_model, metadata_df, index, location_matchers)
        latest_fetch_date = get_latest_fetch_date()
        context = "\n".join(results["text"].tolist())
        prompt = (
            f"Here is a list of French fuel station info fetched on {latest_fetch_date}:\n"
            f"{context}\n\nOriginal question: {user_query}\nEnhanced query: {enhanced_query}\n"
            f"Provide a clear and concise answer based on the enhanced query."
        )
        response = generate_llm_response(prompt, backend=model_choice)

    st.success("✅ Done!")

    # --- Output UI ---
    st.markdown("### 🤖 Chatbot Response")
    if fallback:
        st.info("📌 No exact location match found — showing top semantic results instead.")
    st.markdown(
        f"<div style='padding: 12px; background-color: #f9f9f9; border-left: 4px solid #00aaff; "
        f"white-space: pre-wrap; font-family: monospace; color: #111;'>{html.escape(response)}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### 📄 Context Given to the LLM")
    st.write(f"Fetch Date: {latest_fetch_date}")
    st.markdown(f"**Enhanced Query:** {html.escape(enhanced_query)}")
    st.markdown("---")
    st.code(context, language="text")
else:
    st.warning("⚠️ Please enter a query to search.")
