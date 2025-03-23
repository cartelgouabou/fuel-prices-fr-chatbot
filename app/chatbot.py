import os
import logging
import pickle
import faiss
import torch
import html
import streamlit as st
import numpy as np
from datetime import datetime
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit.watcher.local_sources_watcher as lsw

# Monkey-patch Streamlit to skip torch.classes which causes crashes
original_get_module_paths = lsw.get_module_paths


def safe_get_module_paths(module):
    if getattr(module, "__name__", "") == "torch.classes":
        return []
    try:
        return original_get_module_paths(module)
    except Exception:
        return []


lsw.get_module_paths = safe_get_module_paths


# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== Config ==========
VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"
LOCAL_MODEL_PATH = "./models/tinyllama"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ========== Cached Resources ==========
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_faiss_index_and_metadata():
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(EMBEDDINGS_FILE, "rb") as f:
        metadata_df = pickle.load(f)
    return index, metadata_df


@st.cache_resource
def build_location_matchers(metadata_df):
    def init_matcher(column):
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
def get_latest_fetch_date():
    """Extracts the latest fetch date based on raw_data_*.json filenames."""
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    try:
        files = [f for f in os.listdir(data_path) if f.startswith("raw_data_")]
        dates = [
            datetime.strptime(f.split("_")[-1].replace(".json", ""), "%Y-%m-%d").date()
            for f in files
        ]
        return max(dates) if dates else None
    except Exception:
        return None


def detect_location(query: str, city_matcher, dept_matcher, region_matcher) -> dict:
    q = query.lower()
    return {
        "city": city_matcher.extract_keywords(q)[0]
        if city_matcher.extract_keywords(q)
        else None,
        "department": dept_matcher.extract_keywords(q)[0]
        if dept_matcher.extract_keywords(q)
        else None,
        "region": region_matcher.extract_keywords(q)[0]
        if region_matcher.extract_keywords(q)
        else None,
    }


def search_with_filter(query, embed_model, metadata_df, faiss_index, matchers):
    city_matcher, dept_matcher, region_matcher = matchers
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vec, k=50)

    results = metadata_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    detected = detect_location(query, city_matcher, dept_matcher, region_matcher)

    filtered = results
    if detected["city"]:
        filtered = results[results["city"].str.lower() == detected["city"]]
    elif detected["department"]:
        filtered = results[
            results["code_department"].astype(str) == detected["department"]
        ]
    elif detected["region"]:
        filtered = results[results["region"].str.lower() == detected["region"]]

    fallback = filtered.empty
    return (filtered.head(15) if not fallback else results.head(5)), fallback


def generate_llm_response(prompt, tokenizer, model, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return (
        decoded.split("Answer:", 1)[-1].strip()
        if "Answer:" in decoded
        else decoded.strip()
    )


# ========== Streamlit UI ==========
st.set_page_config(page_title="⛽ Fuel Prices Chatbot", layout="wide")

embed_model = load_embed_model()
tokenizer, llm_model = load_llm_model()
index, metadata_df = load_faiss_index_and_metadata()
location_matchers = build_location_matchers(metadata_df)

# --- Header ---
st.markdown(
    """
    <h1 style='text-align: center;'>⛽ France Fuel Prices Chatbot</h1>
    <p style='text-align: center; font-size: 18px;'>Ask me anything about fuel prices in France, and I’ll use smart retrieval and a local language model to answer!</p>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# --- User Input ---
col1, col2 = st.columns([2, 1])
with col1:
    user_query = st.text_input(
        "💬 What do you want to know?",
        placeholder="e.g. what is the station with the cheapeast sp98 price in Marseille",
    )
    search_clicked = st.button("🔍 Search")
with col2:
    st.image("https://img.icons8.com/color/96/gas-pump.png", width=96)
    st.caption("Powered by MiniLM + TinyLlama + FAISS")

# --- Search + Response ---
if search_clicked and user_query:
    with st.spinner("🔎 Searching and generating response..."):
        results, fallback = search_with_filter(
            user_query, embed_model, metadata_df, index, location_matchers
        )
        latest_fetch_date = get_latest_fetch_date()
        context = "\n".join(results["text"].tolist())
        prompt = f"Given the following fuel stations data (fetched on {latest_fetch_date}):\n{context}\n\nAnswer the user's question concisely: {user_query}\nAnswer:"
        response = generate_llm_response(prompt, tokenizer, llm_model)

    st.success("✅ Done!")

    # Response block
    st.markdown("### 🤖 Chatbot Response")
    if fallback:
        st.info(
            "📌 No exact city/department/region match found — showing top semantic results."
        )
    st.markdown(
        f"""
        <div style='
            padding: 12px;
            background-color: #f9f9f9;
            border-left: 4px solid #00aaff;
            white-space: pre-wrap;
            font-family: monospace;
            color: #111;
        '>{html.escape(response)}</div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Context shown to the LLM
    st.markdown("### 📄 Context Given to the LLM")
    st.write(f"Fetch Date: {latest_fetch_date}")
    st.code(context, language="text")
else:
    st.warning("⚠️ Please enter a query to search.")
