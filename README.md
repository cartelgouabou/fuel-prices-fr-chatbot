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

### 4. Run the ETL pipeline
```bash
(venv-fuel) $ python -m etl.run_etl
```
This performs:
- Extract: Pulls data from `data.gouv.fr` (up to 10k+ records)
- Transform: Cleans and formats into CSV
- Load: Stores into SQLite + FAISS
- Embeds: Generates vector embeddings for chatbot retrieval

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

## 🧑‍💬 Run the Chatbot Interface

```bash
(venv-fuel) $ streamlit run app/chatbot.py
```

- Query in natural language: `Cheapest SP95 near Marseille`
- The app performs semantic search over vector DB and passes top-k context to TinyLlama
- You get smart responses with supporting data shown

---

## ✈️ ETL Pipeline - Step-by-Step

### 1. `fetch_data.py`
Fetches paginated API data from [data.gouv.fr](https://data.gouv.fr) using:
```python
BASE_URL = "https://tabular-api.data.gouv.fr/api/resources/..."
```
Saves results to:
```
data/raw_data_YYYY-MM-DD.json
```

### 2. `transform_data.py`
- Parses raw JSON
- Extracts and cleans fields (fuel types, location, timestamp)
- Saves to:
```
data/processed_data_YYYY-MM-DD.csv
```

### 3. `load_data.py`
- Initializes SQLite database (`fuel_prices.db`)
- Inserts new records

### 4. `process_fuel_embeddings.py`
- Loads latest fuel records from DB
- Formats text for embedding:
  ```text
  Station 12345 in 1 Rue ABC, Paris, 75, Ile-de-France:
  Gazole=1.85, SP95=1.79, ...
  ```
- Encodes text with `MiniLM-L6-v2`
- Indexes into FAISS and saves metadata:
```
data/faiss_index
and
data/embeddings.pkl
```

### 5. `run_etl.py`
One-click pipeline to execute all the above steps in order.

---

## 🤖 Chatbot Flow (app/chatbot.py)

- Load resources at startup: embeddings, FAISS index, LLM, tokenizer
- User enters natural language query
- Semantic search retrieves top-5 similar entries from FAISS
- LLM receives structured context and generates a response

```python
prompt = f"Given the following fuel stations data:\n{context}\n\nAnswer the user's question: {user_query}\nAnswer:"
```

LLM outputs a concise answer which is shown alongside the search context.

---

## ⚡ Models Used

| Purpose             | Model                              |
|--------------------|-------------------------------------|
| Embedding          | `sentence-transformers/all-MiniLM-L6-v2` |
| Local Language Model | `TinyLlama-1.1B-Chat-v1.0`             |
| Vector DB          | FAISS                             |

---

## 🔧 Utilities

- `utils.py`: Helps find the latest processed file by filename timestamp

---

## 🚀 Future Enhancements

- Improve LLM answers with more reasoning
- Add support for map-based search
- Enable GPU acceleration if available
- Add unit tests

---

## 🚫 Disclaimer

This project is for educational/demo purposes. Data is fetched from a public API and may not be real-time.

---

