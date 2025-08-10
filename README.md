# 🌀 Convector – CSV → JSONL (384‑dim Embeddings) + Qdrant Quickstart

## Autor: Roger Camara
### Feel free to contribute.


**Convector** is a tiny, practical toolkit to turn `.csv` datasets into newline‑delimited JSON (`output.jsonl`) with **384‑dim** sentence embeddings and the original row as `payload`. It pairs with a simple importer to load the file into a local **Qdrant** vector DB (Docker).


---

## ✨ What you get

- **convector.py** – reads your CSV, auto‑detects columns, builds one text per row, generates **384‑dim** embeddings, and writes `output.jsonl`:
  ```json
  {"id":"<uuid>", "text":"<row-as-text>", "vector":[...384 floats...], "payload":{...original row...}}
  ```
- **qdrantimport.py** – asks for `output.jsonl`, lists Qdrant collections, and imports in batches with a progress bar.

> We use a **free** embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) that outputs **384 dimensions**. If you switch to another provider (e.g., OpenAI), you can use larger vectors—just make sure your collection size matches.

---

## 📦 Requirements

- Python **3.9+**
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 Convert your CSV → JSONL

1. Put your `.csv` next to `convector.py`.
2. Run:
   ```bash
   python convector.py
   ```
3. Paste/drag your CSV path and confirm the detected columns.
4. You’ll get **`output.jsonl`** in the current folder.

---

## 🧪 Qdrant (Docker) — ultra‑quick setup

### 1) Start Qdrant locally
```bash
docker run -p 6333:6333 --name qdrant --rm qdrant/qdrant
```

### 2) Create a collection with **384** dimensions
```bash
curl -X PUT "http://localhost:6333/collections/my_collection"   -H "Content-Type: application/json"   -d '{"vectors": {"size": 384, "distance": "Cosine"}}'
```

### 3) Import your JSONL
```bash
python qdrantimport.py
```
- Enter `output.jsonl` path.
- Press Enter to keep default Qdrant URL (`http://localhost:6333`).
- Select `my_collection`.
- Watch the progress bar until ✅ Done.

---

## 🔍 Quick query test

### Search by vector
```bash
curl -X POST "http://localhost:6333/collections/my_collection/points/search"   -H "Content-Type: application/json"   -d '{
    "vector": [0.1, 0.2, ... 384 floats ...],
    "limit": 3
  }'
```

### Search by payload filter
```bash
curl -X POST "http://localhost:6333/collections/my_collection/points/scroll"   -H "Content-Type: application/json"   -d '{
    "filter": {
      "must": [
        {"key": "payload.column_name", "match": {"value": "some_value"}}
      ]
    },
    "limit": 3
  }'
```

---

## 📝 Notes

- Output is always `output.jsonl` in the current folder.
- IDs are deterministic (UUID5) for reproducible imports.
- Free model = **384 dimensions**. Change model & collection size together.
