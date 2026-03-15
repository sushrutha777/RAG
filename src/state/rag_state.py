"""RAG state definition for LangGraph"""

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class RAGState(BaseModel):
    """State object for RAG workflow"""
    
    question: str
    messages: List[BaseMessage] = []
    retrieved_docs: List[Document] = []
    decision: str = ""
    confidence: float = 0.0
    answer: str = ""