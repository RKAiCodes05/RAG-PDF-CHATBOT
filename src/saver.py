import json
import sqlite3
import os
from datetime import datetime


class RAGSystemSaver:
    """
    Save and load RAG system state to JSON and SQLite
    
    This class does NOT instantiate RAGSystem during import.
    It only provides methods to save/load when called.
    """
    
    def __init__(self, save_dir="rag_saves"):
        """Initialize save directory"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────
    # JSON Methods
    # ─────────────────────────────────────────────────────────────────
    
    def save_config_json(self, rag_system, name="default"):
        """Save system configuration to JSON"""
        config = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'vector_store': {
                'path': rag_system.vector_store.persist_directory,
                'collection_name': rag_system.vector_store.collection_name,
                'total_documents': rag_system.vector_store.collection.count()
            },
            'embedding': {
                'model_name': rag_system.embedding_manager.model_name,
                'embedding_dim': rag_system.embedding_manager.model.get_sentence_embedding_dimension()
            },
            'llm': {
                'provider': 'groq',
                'default_model': 'llama-3.3-70b-versatile'
            }
        }
        
        filepath = os.path.join(self.save_dir, f"{name}_config.json")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {filepath}")
        return filepath
    
    def load_config_json(self, name="default"):
        """Load system configuration from JSON"""
        filepath = os.path.join(self.save_dir, f"{name}_config.json")
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded from {filepath}")
        return config
    
    # ─────────────────────────────────────────────────────────────────
    # SQLite Methods
    # ─────────────────────────────────────────────────────────────────
    
    def _init_sqlite_db(self, db_path):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT,
                num_contexts INTEGER,
                avg_similarity REAL,
                model_used TEXT,
                session_id TEXT
            )
        ''')
        
        # Sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                source_file TEXT,
                page TEXT,
                similarity REAL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_conversation ON sources(conversation_id)')
        
        conn.commit()
        conn.close()
    
    def save_conversations_sqlite(self, rag_system, name="default", session_id=None):
        """Save conversation history to SQLite database"""
        db_path = os.path.join(self.save_dir, f"{name}_conversations.db")
        
        # Initialize DB
        self._init_sqlite_db(db_path)
        
        # Insert conversations
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        session_id = session_id or datetime.now().isoformat()
        
        for conv in rag_system.conversation_history:
            # Insert conversation
            similarity_scores = [s.get('similarity', 0) for s in conv.get('sources', [])]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            cursor.execute('''
                INSERT INTO conversations 
                (question, answer, num_contexts, avg_similarity, model_used, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conv['question'],
                conv['answer'],
                conv['num_contexts'],
                avg_similarity,
                'llama-3.3-70b-versatile',
                session_id
            ))
            
            conv_id = cursor.lastrowid
            
            # Insert sources
            for source in conv.get('sources', []):
                cursor.execute('''
                    INSERT INTO sources 
                    (conversation_id, source_file, page, similarity)
                    VALUES (?, ?, ?, ?)
                ''', (
                    conv_id,
                    source.get('file'),
                    source.get('page'),
                    source.get('similarity')
                ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(rag_system.conversation_history)} conversations to SQLite")
        print(f"   Database: {db_path}")
    
    def load_conversations_sqlite(self, name="default", limit=20):
        """Load conversations from SQLite database"""
        db_path = os.path.join(self.save_dir, f"{name}_conversations.db")
        
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            return []
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch recent conversations
        cursor.execute('''
            SELECT * FROM conversations 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        conversations = cursor.fetchall()
        
        results = []
        for conv in conversations:
            # Fetch sources
            cursor.execute('''
                SELECT * FROM sources 
                WHERE conversation_id = ?
            ''', (conv['id'],))
            
            sources = cursor.fetchall()
            
            results.append({
                'id': conv['id'],
                'question': conv['question'],
                'answer': conv['answer'],
                'num_contexts': conv['num_contexts'],
                'avg_similarity': conv['avg_similarity'],
                'created_at': conv['created_at'],
                'sources': [
                    {
                        'file': s['source_file'],
                        'page': s['page'],
                        'similarity': s['similarity']
                    }
                    for s in sources
                ]
            })
        
        conn.close()
        print(f"✅ Loaded {len(results)} conversations from SQLite")
        return results
    
    def get_sqlite_stats(self, name="default"):
        """Get statistics from SQLite database"""
        db_path = os.path.join(self.save_dir, f"{name}_conversations.db")
        
        if not os.path.exists(db_path):
            return {}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_convs = cursor.fetchone()[0]
        
        # Average similarity
        cursor.execute('SELECT AVG(avg_similarity) FROM conversations')
        avg_sim = cursor.fetchone()[0] or 0
        
        # Average contexts per query
        cursor.execute('SELECT AVG(num_contexts) FROM conversations')
        avg_contexts = cursor.fetchone()[0] or 0
        
        # Most used sources
        cursor.execute('''
            SELECT source_file, COUNT(*) as count 
            FROM sources 
            GROUP BY source_file 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        top_sources = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_conversations': total_convs,
            'avg_similarity_score': round(avg_sim, 3),
            'avg_contexts_per_query': round(avg_contexts, 2),
            'top_sources': [{'file': s[0], 'count': s[1]} for s in top_sources]
        }
    
    def search_conversations(self, name="default", query_text="", min_similarity=0.0):
        """Search conversations by question/answer content"""
        db_path = os.path.join(self.save_dir, f"{name}_conversations.db")
        
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search with LIKE
        cursor.execute('''
            SELECT * FROM conversations 
            WHERE question LIKE ? OR answer LIKE ?
            AND avg_similarity >= ?
            ORDER BY created_at DESC
        ''', (f"%{query_text}%", f"%{query_text}%", min_similarity))
        
        results = cursor.fetchall()
        conn.close()
        
        print(f"✅ Found {len(results)} matching conversations")
        return results