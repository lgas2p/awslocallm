from typing import List, Optional
from pydantic import BaseModel

# This file is used for containing message models for the API
# Missing relevant in/out messages classes
# Might need to save here enums for settings of indexing and rag

class Message(BaseModel):
    message: str

class ActiveStatus(BaseModel):
    status: bool

class QuestionRequest(BaseModel):
    message: str

class TimeFrames(BaseModel):
    type: str = None
    value: float = None

class Images(BaseModel):
    source: str = None
    type: str = "url"

class Videos(BaseModel):
    source: str = None
    thumb: Images = None

class CompleteOutput(BaseModel):
    bodyText: Optional[str] = None
    bodyHtml: Optional[str] = None
    imgs: Optional[List[Images]] = None
    videos: Optional[List[Videos]] = None
    urls: Optional[List[str]] = None
    timeElapsed: Optional[List[TimeFrames]] = None