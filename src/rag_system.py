from typing import List, Dict, Any
from src.retriever import RAGRetriever
class RAGSystem:
    """Complete RAG system that orchestrates all components"""
    
    def __init__(self, vector_store, embedding_manager, llm_client, retriever=None):
        """
        Initialize RAG system
        
        Args:
            vector_store: Vector store instance
            embedding_manager: Embedding manager instance
            llm_client: LLM client instance
            retriever: Optional pre-initialized retriever
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.retriever = retriever if retriever is not None else RAGRetriever(
            self.vector_store, self.embedding_manager
        )
        self.conversation_history = []
        
        print("✅ RAG System initialized successfully")
    
    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.5, **llm_kwargs):
        """
        Query the RAG system
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity threshold
            **llm_kwargs: Additional arguments for LLM (model, temperature, etc.)
        
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        print(f"\n{'─'*70}")
        print(f"📝 Query: {question}")
        print(f"{'─'*70}\n")
        
        # Retrieve contexts
        contexts = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not contexts:
            print("⚠️ No relevant contexts found")
            response = {
                'answer': 'I could not find relevant information to answer this question.',
                'sources': [],
                'num_contexts_used': 0,
                'question': question
            }
        else:
            # Generate response
            response = self.llm_client.generate_response(
                query=question,
                contexts=contexts,
                **llm_kwargs
            )
            response['question'] = question
            response['retrieved_contexts'] = contexts
        
        # Track conversation
        self.conversation_history.append({
            'question': question,
            'answer': response.get('answer'),
            'sources': response.get('sources', []),
            'num_contexts': response.get('num_contexts_used', 0)
        })
        
        return response
    
    def display_response(self, response: dict):
        """Display response in a formatted way"""
        print("\n" + "="*70)
        print("✅ ANSWER")
        print("="*70)
        print(response.get('answer', 'No answer returned'))
        
        print("\n" + "="*70)
        print(f"📄 SOURCES ({response.get('num_contexts_used', 0)} documents)")
        print("="*70)
        
        sources = response.get('sources', [])
        if not sources:
            print("  No sources available")
        else:
            for src in sources:
                if isinstance(src, dict):
                    print(f"  [{src.get('id', '?')}] {src.get('file', 'Unknown')} "
                          f"(Page {src.get('page', 'N/A')}) - "
                          f"Relevance: {src.get('similarity', 0.0):.1%}")
                else:
                    print(f"  - {src}")
        
        print(f"\n🤖 Model: {response.get('model', 'N/A')} | "
              f"Contexts: {response.get('num_contexts_used', 0)}")
        print("="*70 + "\n")
    
    def get_stats(self):
        """Get session statistics"""
        if not self.conversation_history:
            return {
                "total_queries": 0,
                "total_contexts_retrieved": 0,
                "avg_contexts_per_query": 0.0
            }
        
        total_contexts = sum(c['num_contexts'] for c in self.conversation_history)
        return {
            "total_queries": len(self.conversation_history),
            "total_contexts_retrieved": total_contexts,
            "avg_contexts_per_query": total_contexts / len(self.conversation_history)
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")