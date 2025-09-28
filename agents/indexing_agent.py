# agents/indexing_agent.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# configure DB connection (same fallback logic)
PG_HOST = os.getenv("POSTGRES_HOST") or os.getenv("MYSQL_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT") or os.getenv("MYSQL_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER") or os.getenv("MYSQL_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD") or os.getenv("MYSQL_PASSWORD", "")
PG_DB = os.getenv("POSTGRES_DB") or os.getenv("MYSQL_DATABASE") or "argo_db"

PG_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(PG_URL, pool_pre_ping=True)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # for all-MiniLM-L6-v2
MODEL = SentenceTransformer(MODEL_NAME)

# FAISS files
STORE_DIR = os.getenv("FAISS_STORE_DIR", "faiss_store")
os.makedirs(STORE_DIR, exist_ok=True)
INDEX_FILE = os.path.join(STORE_DIR, "faiss.index")
METADATA_FILE = os.path.join(STORE_DIR, "metadata.pkl")

def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        # create new index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # inner product for cosine (after normalization)
        metadata = []  # list of dicts with fields: id (int), text, source
        return index, metadata

def save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def embed_texts(texts):
    embs = MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize for cosine similarity
    faiss.normalize_L2(embs)
    return embs

def index_records(records):
    """
    records: list of dicts { 'id': <int or str>, 'text': <str>, 'source': <str> }
    We append them to FAISS and metadata store. 'id' should be unique.
    """
    index, metadata = load_index()
    texts = [r["text"] for r in records]
    embs = embed_texts(texts)
    start_id = len(metadata)
    index.add(embs)
    for i, r in enumerate(records):
        metadata.append({"global_id": start_id + i, "orig_id": r.get("id"), "text": r["text"], "source": r.get("source")})
    save_index(index, metadata)
    return True

def query_similar(query_text, top_k=5):
    index, metadata = load_index()
    if index.ntotal == 0:
        return []
    q_emb = embed_texts([query_text])
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({"score": float(dist), "text": meta["text"], "source": meta["source"], "global_id": meta["global_id"]})
    return results
