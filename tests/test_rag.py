import pytest
import numpy as np
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
# from src.rag_system import RAGSystem # Requires more mocks, let's test components first

def test_embedding_generation():
    """Test that embeddings are generated with correct shape"""
    mgr = EmbeddingManager()
    # Test with a simple string
    text = "hello world"
    embeddings = mgr.generate_embeddings([text])
    
    # Check shape (1, 384) for all-MiniLM-L6-v2
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 384

def test_vector_store_initialization():
    """Test vector store inits correctly"""
    store = VectorStore(embedding_dim=384)
    assert store.index.ntotal == 0
    assert len(store.documents) == 0

def test_vector_store_add_search():
    """Test adding documents and searching"""
    store = VectorStore(embedding_dim=384)
    
    # Mock data
    dummy_embedding = np.random.rand(1, 384).astype("float32")
    class MockDoc:
        page_content = "test content"
        metadata = {"source": "test.pdf"}
    
    store.add_documents([MockDoc()], dummy_embedding)
    
    assert store.index.ntotal == 1
    
    # Search
    results = store.search(dummy_embedding)
    assert len(results) == 1
    assert results[0]['content'] == "test content"
