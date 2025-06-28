# ProtonPDF
ProtonPDF is a PDF reader that lets users upload, view, and chat using an AI chatbot. Powered by React tools and ~~TinyLlama/TinyLlama-1.1B-Chat-v1.0~~ ~~~meta-llama/llama-4-scout-17b-16e-instruct LLM using GROQ_API_KEY~~~ Google gemini using GOOGLE_API_KEY , it contextualizes the understanding of documents without reading them fully.


### API instances:
```
GOOGLE_API_KEY 
REDIS_URL
```
## Features

- Upload any PDF document
- View documents using pdf-reader 
- Light-weight questions with an AI model.


## Tech Stack

### Frontend
- ReactJS + Tailwind CSS
- React-PDF for rendering documents
- Axios for API calls

### Backend
- Python (FastAPI / Flask)
- Redis (Sub/Pub for real-time requests)
- FAISS for vec search
- ~~TinyLlama a very light weight model for chat (can be updated later)~~~
- gemini-1.5-flash using the GOOGLE_API_KEY

## Folder Structure 
    backend/
    ├── __pycache__/
    ├── models/
    ├── uploads/
    ├── vectorstores/
    ├── venv/
    ├── .gitignore
    ├── main.py
    ├── qa.py
    ├── t.py
    └── worker.py



## Install Dependencies 
pip install -r requirements.txt

## To run the App
npm start

## To access the server
Navigate to 
```
http://127.0.0.1:8000/docs
```
in your web browser or any api client

~~``"Can,
Integrate a larger LLM (e.g., LLaMA 3)"``~~
``DONE``

