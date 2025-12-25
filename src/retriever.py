from .vector_store import VectorStore
from .embeddings import EmbeddingManager
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        if len(query) > 2000:
            print("⚠️ Query too long, truncating to 2000 characters")
            query = query[:2000]

        print(f"🔍 Retrieving documents for query: '{query[:50]}...'")
        print(f"   Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # FAISS search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Apply score threshold
        filtered_results = [
            r for r in results
            if r["similarity_score"] >= score_threshold
        ]
        
        print(f"✅ Retrieved {len(filtered_results)} documents")
        return filtered_results
