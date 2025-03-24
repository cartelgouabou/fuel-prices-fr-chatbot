import os
import logging
import pickle
import faiss
import torch
import numpy as np
from datetime import datetime
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import chainlit as cl

# ========== Config ==========
VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDINGS_FILE = "data/embeddings.pkl"
LOCAL_MODEL_PATH = "./models/tinyllama"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Resources ==========
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
model.eval()

index = faiss.read_index(VECTOR_STORE_PATH)
with open(EMBEDDINGS_FILE, "rb") as f:
    metadata_df = pickle.load(f)

# ========== Location Matchers ==========
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

location_matchers = build_location_matchers(metadata_df)

# ========== Utility Functions ==========
def get_latest_fetch_date():
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
        "city": city_matcher.extract_keywords(q)[0] if city_matcher.extract_keywords(q) else None,
        "department": dept_matcher.extract_keywords(q)[0] if dept_matcher.extract_keywords(q) else None,
        "region": region_matcher.extract_keywords(q)[0] if region_matcher.extract_keywords(q) else None,
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
        filtered = results[results["code_department"].astype(str) == detected["department"]]
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
    return decoded.split("Answer:", 1)[-1].strip() if "Answer:" in decoded else decoded.strip()

# ========== Chainlit Logic ==========
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("embed_model", embed_model)
    cl.user_session.set("tokenizer", tokenizer)
    cl.user_session.set("model", model)
    cl.user_session.set("index", index)
    cl.user_session.set("metadata_df", metadata_df)
    cl.user_session.set("location_matchers", location_matchers)

    await cl.Message(author="assistant", content="""
⛽️ **Welcome to the France Fuel Prices Chatbot!**

Ask me anything about fuel prices in France, and I'll help you find the info using smart search and a local language model.

Example: *What is the station with the cheapest SP98 price in Marseille?*
""").send()

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    embed_model = cl.user_session.get("embed_model")
    tokenizer = cl.user_session.get("tokenizer")
    model = cl.user_session.get("model")
    index = cl.user_session.get("index")
    metadata_df = cl.user_session.get("metadata_df")
    location_matchers = cl.user_session.get("location_matchers")

    results, fallback = search_with_filter(query, embed_model, metadata_df, index, location_matchers)
    latest_fetch_date = get_latest_fetch_date()
    context = "\n".join(results["text"].tolist())
    prompt = f"Given the following fuel stations data (fetched on {latest_fetch_date} and the prices in EUR):\n{context}\n\nAnswer the user's question concisely: {query}\nAnswer:"
    response = generate_llm_response(prompt, tokenizer, model)

    if fallback:
        await cl.Message(content="📌 No exact city/department/region match found — showing top semantic results.").send()

    await cl.Message(content=response).send()

    await cl.Message(
        author="system",
        content=f"**Context used (fetched on {latest_fetch_date}):**",
        language="text",
        elements=[
            cl.Text(name="Fuel Station Context", content=context, display="inline")
        ]
    ).send()
