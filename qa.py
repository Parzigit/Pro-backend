import pickle, faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer 

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def answer_question(job_id, question):
    with open(f"./vectorstores/{job_id}.pkl", "rb") as f:
        chunks, index = pickle.load(f)
    question_embedding = embedder.encode([question])
    top_k = 3
    D, I = index.search(question_embedding, top_k)
    context_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQ: {question}\nA:"
    
    result = generator(prompt, max_new_tokens=100)[0]["generated_text"]
    return result.split("A:")[-1].strip()
