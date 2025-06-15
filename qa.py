import os
from groq import Groq
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
try:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set!")
    client = Groq(api_key=api_key)
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    raise

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def answer_question(job_id, question):
    try:
        with open(f"./vectorstores/{job_id}.pkl", "rb") as f:
            chunks, index = pickle.load(f)
        question_embedding = embedder.encode([question])
        top_k = 5
        D, I = index.search(question_embedding, top_k)
        context_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(context_chunks)
        prompt = f"""
Context:
{context}

Question: {question}
Answer the question based on the pdf uploade. IF mentioned about the number of lines to be uploaded provide them with that many number of lines of content and also consider when user is speaking about pdf they mean document.,
You are iterating back and forth with a user on their request.,
Aim to fulfill the user's request with minimal back-and-forth interactions.,
You're a helpful assistant designed to quickly explore and clarify user questions strictly related to the uploaded pdf., 
Give careful considerations towards what the user is asking for,i.e, user may ask straightforwad questions like who the pdf is being presented to or who are the target audience for this pdf, etcetera..Give conscise and accurate answers.,
Provide Single word answers if needed for straightforward questions. 
"""
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        answer = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            answer += content
        return answer
    except Exception as e:
        print(f"Error in answer_question: {e}")
        raise
