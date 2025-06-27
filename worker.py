import os
import fitz 
import faiss
import pickle
from redis import Redis
from rq import Queue, Connection, Worker
from sentence_transformers import SentenceTransformer

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(redis_url)

model = SentenceTransformer('all-MiniLM-L6-v2')

def process_pdf(path, file_id):
    with fitz.open(path) as doc:
        text = "\n".join([page.get_text() for page in doc])
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("No extractable text in PDF.")
    chunks = [text[i:i+250] for i in range(0, len(text), 250)]
    embeddings = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs("./vectorstores", exist_ok=True)
    with open(f"./vectorstores/{file_id}.pkl", "wb") as f:
        pickle.dump((chunks, index), f)
    return "done"

if __name__ == '__main__':
    with Connection(redis_conn):
        q = Queue('pdf-tasks', connection=redis_conn)
        worker = Worker([q], connection=redis_conn)
        worker.work()
