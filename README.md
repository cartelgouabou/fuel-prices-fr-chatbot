# 🚗 fuel-prices-fr-chatbot

A fully functional end-to-end ETL + Chatbot pipeline to track and explore **fuel prices in France**, combining:

- ⚡ ETL pipeline (Extract, Transform, Load)
- 🧠 Semantic search (SentenceTransformers + FAISS)
- 🧬 Local LLM (TinyLlama)
- 🛍️ Interactive chatbot interface (Streamlit)

---

## ♻️ Project Structure

```
fuel-prices-fr-chatbot/
├── app/                    # Streamlit chatbot app
│   └── chatbot.py
├── data/                  # Data storage (raw, processed, embeddings, DB)
│   ├── raw_data_YYYY-MM-DD.json
│   ├── processed_data_YYYY-MM-DD.csv
│   ├── fuel_prices.db
│   ├── embeddings.pkl
│   └── faiss_index
├── etl/                   # ETL pipeline scripts
│   ├── fetch_data.py
│   ├── transform_data.py
│   ├── load_data.py
│   ├── process_fuel_embeddings.py
│   └── run_etl.py
├── models/                # Local LLM models (TinyLlama)
│   └── tinyllama/
├── utilities/             # Utility functions
│   └── utils.py
├── download_model.sh      # Script to download TinyLlama
├── requirements.txt       # Python dependencies
├── fuel_data_processing.log
├── README.md
└── tests/                 # (Optional) Tests folder
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo and create virtual environment
```bash
$ git clone https://github.com/cartelgouabou/fuel-prices-fr-chatbot.git
$ cd fuel-prices-fr-chatbot
$ python -m venv venv-fuel
$ source venv-fuel/bin/activate  # On Windows: venv-fuel\Scripts\activate
```

### 2. Install dependencies
```bash
(venv-fuel) $ pip install -r requirements.txt
```

### 3. Download the TinyLlama model (locally)
```bash
(venv-fuel) $ chmod +x download_model.sh
(venv-fuel) $ ./download_model.sh
```

## ⚙️ ETL Pipeline Overview

The ETL pipeline extracts and transforms fuel data, loads it into a local database, and prepares it for semantic retrieval with embeddings.

```bash
python -m etl.run_etl
```

This runs the following scripts:

### 1. `fetch_data.py` — **Extract**
Pulls paginated JSON data via the public API:
```text
https://tabular-api.data.gouv.fr/api/resources/...
```
Stores to:
```
data/raw_data_YYYY-MM-DD.json
```

![API Screenshot](assets/api_illustration.png)

### 2. `transform_data.py` — **Transform**
- Cleans up null values, formats fuel types, location, timestamps
- Saves structured CSV to:
```
data/processed_data_YYYY-MM-DD.csv
```

### 3. `load_data.py` — **Load**
- Initializes the SQLite DB (`fuel_prices.db`)
- Inserts transformed rows into a relational schema

### 4. `process_fuel_embeddings.py` — **Embed** ✅ **(Key RAG Step)**

This script is critical to enabling RAG (Retrieval-Augmented Generation). It does:

- **Query Latest Data**: Uses SQL with `MAX(updated_at)` to get the latest fuel prices for each station.

- **Format Text for Embedding**:
```text
Station 12345 in 1 Rue ABC, Paris, 75, Ile-de-France:
Gazole=1.85, SP95=1.79, ...
```

- **Embed with MiniLM**:
```python
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), convert_to_numpy=True)
```

- **Index with FAISS**:
```python
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

- **Save**:
  - `data/faiss_index`
  - `data/embeddings.pkl` (contains original metadata)

> This step connects raw structured data with semantic retrieval by enabling fast similarity search over station descriptions.

### 5. `run_etl.py` — **Pipeline Orchestrator**
Handles all steps in order with logging and error handling.

---


## 🤖 Chatbot Flow (app/chatbot.py)

### Step-by-Step RAG Pipeline

1. **User Input**
   > e.g. "Cheapest SP95 near Lyon"

2. **Semantic Search (FAISS)**
```python
results = find_similar_stations(user_query, index, metadata_df, embed_model)
```
Returns top-5 results from vector index with metadata.

3. **Prompt Building**
Combines user query and retrieved context into a single input for the LLM:
```python
prompt = f"Given the following fuel stations data:\n{context}\n\nAnswer the user's question: {user_query}\nAnswer:"
```

4. **LLM Generation with TinyLlama**
```python
response = generate_llm_response(prompt, tokenizer, model)
```
Uses local `TinyLlama-1.1B-Chat-v1.0` in FP16 (if GPU is available).

5. **Display Answer and Supporting Context**
- Final answer is shown in a styled output box
- Context is printed below to ensure transparency

![Chatbot Screenshot](assets/response.png)

---

### 🔍 Enhancing Context Filtering with FlashText

To improve location-aware filtering in responses, the chatbot includes logic using `flashtext.KeywordProcessor` to detect **city**, **department**, or **region** keywords in user queries.

```python
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
```

- **detect_location()**: Identifies the most likely matching location from the query.
- **search_with_filter()**: Applies a priority filter based on detected locations, improving relevance in FAISS matches without re-embedding.
- This enhances semantic retrieval by combining vector similarity with keyword-based filtering.

---
## 🌐 Run the Chatbot Interface
![Chatbot Screenshot](assets/app.png)

```bash
streamlit run app/chatbot.py
```

You can ask:
> "Where is the cheapest E10 in Marseille?"

---

## 🚧 Automating ETL with a Scheduled Job

To ensure your fuel price data stays updated daily, you can schedule the ETL pipeline (`run_etl.py`) using:

### ✅ Linux/macOS (using `cron`)

1. Open crontab editor:
```bash
crontab -e
```

2. Add the following line to run the ETL script every day at 3am:
```bash
0 3 * * * /path/to/venv-fuel/bin/python /path/to/fuel-prices-fr-chatbot/etl/run_etl.py >> /path/to/fuel-prices-fr-chatbot/logs/etl_cron.log 2>&1
```

- Replace `/path/to/venv-fuel` and project path accordingly
- Logs will be saved to `logs/etl_cron.log` (create the `logs/` folder if needed)

### ✅ Windows (using Task Scheduler)

1. Open Task Scheduler → Create Basic Task
2. Set schedule (e.g. Daily at 3am)
3. For action, select: **Start a program**
4. Use `python.exe` from your virtual env and pass full path to `run_etl.py`:
```
Program/script:
    C:\path\to\venv-fuel\Scripts\python.exe
Add arguments:
    C:\path\to\fuel-prices-fr-chatbot\etl\run_etl.py
```

Ensure that the task is allowed to run with the correct user permissions.

---