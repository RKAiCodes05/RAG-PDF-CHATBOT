import json
import sqlite3
import os
import faiss
from datetime import datetime

class RAGSystemSaver:
    """Save/load FAISS-based RAG system"""
    
    def __init__(self, save_dir="rag_saves"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_all(self, rag_system, name="default"):
        timestamp = datetime.now().isoformat()
        base_path = os.path.join(self.save_dir, name)
        os.makedirs(base_path, exist_ok=True)
        
        # 1️⃣ Save FAISS index
        index_path = f"{base_path}/faiss.index"
        faiss.write_index(rag_system.vector_store.index, index_path)
        
        # 2️⃣ Save documents + metadata
        docs_path = f"{base_path}/documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": rag_system.vector_store.documents,
                    "metadatas": rag_system.vector_store.metadatas
                },
                f,
                indent=2
            )
        
        # 3️⃣ Save config
        config = {
            "name": name,
            "created_at": timestamp,
            "embedding_model": rag_system.embedding_manager.model_name,
            "embedding_dim": rag_system.vector_store.embedding_dim,
            "total_documents": len(rag_system.vector_store.documents)
        }
        
        with open(f"{base_path}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # 4️⃣ Save conversations to SQLite
        db_path = f"{base_path}/conversations.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                num_contexts INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for conv in rag_system.conversation_history:
            cursor.execute("""
                INSERT INTO conversations (question, answer, num_contexts)
                VALUES (?, ?, ?)
            """, (
                conv["question"],
                conv["answer"],
                conv["num_contexts"]
            ))
        
        conn.commit()
        conn.close()
        
        print("✅ RAG system saved successfully")
        print(f"   Path: {base_path}")
        print(f"   FAISS index: faiss.index")
        print(f"   Documents: documents.json")
        print(f"   Conversations: conversations.db")
