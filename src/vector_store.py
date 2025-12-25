import faiss
import numpy as np
import os
import json
class VectorStore:
    """Manages document embeddings using FAISS"""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
        self.documents = []
        self.metadatas = []
        print("✅ FAISS vector store initialized")
    
    def add_documents(self, documents, embeddings):
        """
        Add documents and embeddings to FAISS index
        """
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)  # required for cosine similarity
        
        self.index.add(embeddings)
        
        for doc in documents:
            self.documents.append(doc.page_content)
            self.metadatas.append(doc.metadata)
        
        print(f"✅ Added {len(documents)} documents to FAISS index")
        print(f"   Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding, top_k=5):
        """
        Search FAISS index
        """
        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            results.append({
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "similarity_score": float(distances[0][rank]),
                "rank": rank + 1
            })
        
        return results

    def save(self, path="data/faiss_store"):
            os.makedirs(path, exist_ok=True)

            faiss.write_index(self.index, os.path.join(path, "faiss.index"))

            with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
                json.dump(self.documents, f)

            with open(os.path.join(path, "metadatas.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadatas, f)

    def load(self, path="data/faiss_store"):
            index_path = os.path.join(path, "faiss.index")
            if not os.path.exists(index_path):
                return False

            self.index = faiss.read_index(index_path)

            with open(os.path.join(path, "documents.json"), "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            with open(os.path.join(path, "metadatas.json"), "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)

            return True
