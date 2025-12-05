"""
RAG System - Retrieval-Augmented Generation for PDFs

A production-grade system for semantic search and question answering
over PDF documents.
"""

__version__ = "1.0.0"
__author__ = "RAM"

# Import all your classes
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.llm_client import GroqClient
from src.rag_system import RAGSystem
from src.saver import RAGSystemSaver
from src.document_loader import process_all_pdfs
from src.text_processor import split_documents

# Define what users can import
__all__ = [
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
    "GroqClient",
    "RAGSystem",
    "RAGSystemSaver",
    "process_all_pdfs",
    "split_documents",
]
