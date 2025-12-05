from typing import List, Dict, Any
import os
from groq import Groq

class GroqClient:
    """Client to interact with Groq LLM API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Groq client
        
        Args:
            api_key: Groq API key (if None, reads from GROQ_API_KEY env variable)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable or pass api_key parameter."
            )
        
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client with API key"""
        try:
            print("Initializing Groq client...")
            client = Groq(api_key=self.api_key)
            print("✅ Groq client initialized successfully")
            return client
        except Exception as e:
            print(f"❌ Error initializing Groq client: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved contexts
        
        Args:
            query: User's question
            contexts: List of retrieved documents from RAGRetriever
            model: Groq model to use
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
        
        Returns:
            Dictionary with answer and metadata
        """
        if not contexts:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "model": model,
                "num_contexts_used": 0,
                "avg_similarity": 0.0
            }
        
        # Format contexts with numbering for citations
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx['metadata'].get('source_file', 'Unknown')
            page = ctx['metadata'].get('page', 'N/A')
            similarity = ctx.get('similarity_score', 0.0)
            
            formatted_contexts.append(
                f"[{i}] (Source: {source}, Page: {page}, Relevance: {similarity:.1%})\n{ctx['content']}"
            )
        
        context_text = "\n\n".join(formatted_contexts)
        
        # Create structured prompt
        prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context from PDF documents.

IMPORTANT INSTRUCTIONS:
1. Only use information from the context below
2. Cite sources using [1], [2], [3] format after relevant statements
3. If the context doesn't contain enough information, clearly state that
4. Be concise but comprehensive
5. If you're uncertain, express appropriate confidence levels

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (with citations):"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Extract source information
            sources = [
                {
                    "id": i + 1,
                    "file": ctx['metadata'].get('source_file', 'Unknown'),
                    "page": ctx['metadata'].get('page', 'N/A'),
                    "similarity": ctx.get('similarity_score', 0.0)
                }
                for i, ctx in enumerate(contexts)
            ]
            
            # Calculate average similarity
            avg_similarity = sum(s['similarity'] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "answer": answer,
                "sources": sources,
                "model": model,
                "num_contexts_used": len(contexts),
                "avg_similarity": avg_similarity
            }
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "model": model,
                "num_contexts_used": 0,
                "avg_similarity": 0.0
            }