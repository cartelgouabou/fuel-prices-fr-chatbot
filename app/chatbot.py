import streamlit as st  # Streamlit for building the web UI
import faiss  # Facebook AI Similarity Search for fast vector search
import pickle  # For loading serialized Python objects (e.g., metadata)
from sentence_transformers import (
    SentenceTransformer,
)  # Pretrained sentence embedding model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)  # For loading the local language model
import torch  # PyTorch for model inference
import logging  # Logging utility
import html  # For escaping HTML
import streamlit.watcher.local_sources_watcher as lsw  # Streamlit's file-watching internals (for patching crash)

device = "cuda" if torch.cuda.is_available() else "cpu"
# ==== Monkey-patch to avoid Streamlit crashing when inspecting torch.classes ====
def safe_get_module_paths(module):
    """
    Safely extract module paths while ignoring problematic modules like `torch.classes`
    that can crash Streamlit's file watcher.
    """
    from streamlit.watcher import util

    try:
        # Only process modules that have a __name__ and __path__
        if not hasattr(module, "__name__") or not hasattr(module, "__path__"):
            return []
        # Skip the problematic `torch.classes` module
        if module.__name__ == "torch.classes":
            return []
        # Return paths that are safe to watch
        return [p for p in module.__path__ if util.is_file_watched(p)]
    except Exception:
        return []


# Apply the patch to Streamlit's file watcher
lsw.get_module_paths = safe_get_module_paths

# ==== Logging Configuration ====
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==== Paths ====
VECTOR_STORE_PATH = "data/faiss_index"  # Path to FAISS vector index
EMBEDDINGS_FILE = (
    "data/embeddings.pkl"  # Path to pickled metadata aligned with embeddings
)
LOCAL_MODEL_PATH = "./models/tinyllama"  # Path to local TinyLlama model (HF format)


# ==== Cached Resource: Sentence Embedding Model ====
@st.cache_resource
# --------------------------------------------
# @st.cache_resource
# --------------------------------------------
# This decorator tells Streamlit to cache the *resource* returned by this function
# (such as a model or index), so it's only loaded once per session.
# Unlike @st.cache_data (which is for immutable data like DataFrames),
# @st.cache_resource is used for heavier, stateful objects like:
# - HuggingFace models
# - FAISS indexes
# - Tokenizers
# - PyTorch models
#
# This makes the app much faster when rerunning or updating widgets,
# because these expensive operations don’t repeat unnecessarily.
# --------------------------------------------
def load_embed_model():
    """Loads the MiniLM embedding model from Hugging Face."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ==== Cached Resource: Local LLM Model ====
@st.cache_resource
def load_llm_model():
    """
    Loads a local causal language model (TinyLlama) and tokenizer.
    Model is set to eval mode and uses float32 precision.
    """
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()  # Set model to inference mode
    return tokenizer, model


# ==== Cached Resource: FAISS Index & Metadata ====
@st.cache_resource
def load_faiss_index_and_metadata():
    """
    Loads the FAISS index used for similarity search and the associated metadata
    (which maps embedding vectors back to real-world fuel station info).
    """
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(EMBEDDINGS_FILE, "rb") as f:
        metadata_df = pickle.load(f)
    return index, metadata_df


# ==== Core Logic: Find Similar Stations ====
def find_similar_stations(query, index, metadata_df, embed_model, k=5):
    """
    Encodes a query into a vector, searches the FAISS index,
    and returns the top-k most similar fuel stations with metadata.
    """
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)  # FAISS search
    results = metadata_df.iloc[indices[0]].copy()  # Retrieve metadata rows by index
    results["distance"] = distances[0]  # Add distance (relevance) scores
    return results


# ==== Core Logic: Generate LLM Response ====
def generate_llm_response(prompt, tokenizer, model, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad(): # Disable gradient calculation thus reducing memory usage
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract only the part after "Answer:"
    if "Answer:" in full_response:
        return full_response.split("Answer:", 1)[1].strip()
    else:
        return full_response.strip()



# ==== Streamlit UI ====
st.set_page_config(page_title="⛽ Fuel Prices Chatbot", layout="wide")

# 🚀 Load models and data at startup
embed_model = load_embed_model()
tokenizer, llm_model = load_llm_model()
index, metadata_df = load_faiss_index_and_metadata()

# UI title and layout
st.markdown(
    """
    <h1 style='text-align: center;'>⛽ France Fuel Prices Chatbot</h1>
    <p style='text-align: center; font-size: 18px;'>Ask me anything about fuel prices in France, and I’ll use smart retrieval and a local language model to answer!</p>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# Split layout: Input on left, illustration/help on right
col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input(
        "💬 What do you want to know?", placeholder="e.g. Cheapest SP95 near Lyon"
    )
    search_clicked = st.button("🔍 Search")

with col2:
    st.image("https://img.icons8.com/color/96/gas-pump.png", width=96)
    st.caption("Powered by MiniLM + TinyLlama + FAISS")

# Run search logic
if search_clicked and user_query:
        with st.spinner("🔎 Searching and generating response..."):
            results = find_similar_stations(user_query, index, metadata_df, embed_model)
            
            # Assemble the context passed to the LLM
            context = "\n".join(results["text"].tolist())
            prompt = f"Given the following fuel stations data:\n{context}\n\nAnswer the user's question concisely: {user_query}\nAnswer:"
            response = generate_llm_response(prompt, tokenizer, llm_model)

        st.success("✅ Done!")

        # === Display only the generated response ===
        st.markdown("### 🤖 Chatbot Response")
        escaped_response = html.escape(response)
        st.markdown(f"""
        <div style='
            padding: 12px;
            background-color: #f9f9f9;
            border-left: 4px solid #00aaff;
            white-space: pre-wrap;
            font-family: monospace;
            color: #111;
        '>
        {escaped_response}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # === Show raw context string used in LLM prompt ===
        st.markdown("### 📄 Context Given to the LLM")
        st.code(context, language="text")

else:
        st.warning("⚠️ Please enter a query to search.")

