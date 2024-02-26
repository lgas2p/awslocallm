from fastapi import Body, FastAPI
from mock import LocalLLM
from models import (
    CompleteOutput,
    ActiveStatus,
    QuestionRequest
)
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["content_type"],
)

mock = LocalLLM()

@app.get("/", include_in_schema=False)
def read_root():
    return {"Hello": "World"}

@app.get("/rag/status", response_model=ActiveStatus)
def rag_status():
    return {"status": mock.status}

@app.post("/rag/run", response_model=CompleteOutput)
async def rag_request(body: QuestionRequest = Body(...)):
    results = await asyncio.gather(mock.execute(body))
    return results[0]