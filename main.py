from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from redis import Redis
from rq import Queue
import uuid, tempfile, shutil, os
from worker import process_pdf
from qa import answer_question 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(redis_url)
task_queue = Queue('pdf-tasks', connection=redis_conn)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()  
        job = task_queue.enqueue(process_pdf, tmp.name, file_id)
    return {"job_id": file_id}

@app.get("/ask")
async def ask_question(job_id: str, question: str):
    try:
        answer = answer_question(job_id, question) 
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
@app.get("/pdf/{job_id}")
def serve_pdf(job_id: str):
    return FileResponse(f"./uploads/{job_id}.pdf", media_type='application/pdf')
