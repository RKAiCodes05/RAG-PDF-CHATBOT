import streamlit as st
import os
import json
import csv
import time
from datetime import datetime
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================================

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* General Settings */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            color: #6B7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Chat Message Styling */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Dark mode adjustments for sidebar if needed */
        @media (prefers-color-scheme: dark) {
            section[data-testid="stSidebar"] {
                background-color: #1e293b;
                border-right: 1px solid #334155;
            }
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #4F46E5;
        }

        /* Custom Buttons */
        .stButton button {
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Source Cards */
        .source-card {
            background-color: #f3f4f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #4F46E5;
        }
        
        @media (prefers-color-scheme: dark) {
            .source-card {
                background-color: #1f2937;
                border-left: 4px solid #818cf8;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "index_stats" not in st.session_state:
    st.session_state.index_stats = {
        "pdfs": 0,
        "chunks": 0,
        "vectors": 0
    }

def build_index_once(rag_system):
    if st.session_state.get("rag_index_built", False):
        return

    # Try loading existing FAISS index first
    loaded = rag_system.vector_store.load()
    if loaded:
        st.session_state.rag_index_built = True
        st.success(
            f"✅ Loaded FAISS index ({rag_system.vector_store.index.ntotal} vectors)"
        )
        return

    from src.document_loader import process_all_pdfs
    from src.text_processor import split_documents

    st.info("📚 Building FAISS index (first time only)...")

    docs = process_all_pdfs("data/pdf")
    chunks = split_documents(docs)

    texts = [c.page_content for c in chunks]
    total = len(texts)

    # 🔽 PROGRESS BAR STARTS HERE
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    batch_size = 32
    all_embeddings = []

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]

        batch_embeddings = rag_system.embedding_manager.generate_embeddings(
            batch_texts
        )

        all_embeddings.extend(batch_embeddings)

        progress_bar.progress(min((i + batch_size) / total, 1.0))
        status_text.text(f"Embedding chunks {min(i + batch_size, total)} / {total}")

    progress_bar.empty()
    status_text.empty()
    # 🔼 PROGRESS BAR ENDS HERE

    import numpy as np

    all_embeddings = np.array(all_embeddings, dtype="float32")
    rag_system.vector_store.add_documents(chunks, all_embeddings)


    # Save FAISS to disk
    rag_system.vector_store.save()

    st.session_state.rag_index_built = True
    st.success(
        f"✅ Indexed & saved {rag_system.vector_store.index.ntotal} chunks"
    )


# ============================================================================
# RAG SYSTEM LOADING (FAST – NO INDEXING HERE)
# ============================================================================

@st.cache_resource
def get_rag_system():
    """
    Fast RAG system initializer.
    Does NOT load PDFs or build FAISS index.
    """
    from src.rag_system import RAGSystem
    from src.vector_store import VectorStore
    from src.embeddings import EmbeddingManager
    from src.llm_client import GroqClient

    # 1️⃣ Embedding model (heavy but cached)
    embedding_manager = EmbeddingManager()

    # 2️⃣ FAISS vector store (empty for now)
    embedding_dim = embedding_manager.model.get_sentence_embedding_dimension()
    vector_store = VectorStore(embedding_dim=embedding_dim)

    # 3️⃣ LLM client
    llm_client = GroqClient()

    # 4️⃣ RAG system (empty index)
    rag = RAGSystem(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        llm_client=llm_client
    )

    return rag
rag_system = get_rag_system()

if rag_system:
    build_index_once(rag_system)





# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def prepare_export_data():
    """Convert message history to exportable format"""
    export_data = []
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            # Find the corresponding user question (the message before this one)
            # This assumes a strict user-assistant turn-taking
            idx = st.session_state.messages.index(msg)
            question = st.session_state.messages[idx-1]["content"] if idx > 0 else "Unknown"
            
            export_data.append({
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "question": question,
                "answer": msg["content"],
                "sources": msg.get("sources", []),
                "metrics": msg.get("metrics", {})
            })
    return export_data

def export_json(data):
    return json.dumps({"exported_at": datetime.now().isoformat(), "conversations": data}, indent=2)

def export_csv(data):
    if not data: return ""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Question", "Answer", "Num Sources", "Avg Confidence"])
    for item in data:
        writer.writerow([
            item["timestamp"],
            item["question"],
            item["answer"],
            len(item["sources"]),
            f"{item['metrics'].get('confidence', 0):.2%}"
        ])
    return output.getvalue()

with st.sidebar:
    # App Branding
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <span style='font-size: 2.5rem;'>🧠</span>
            <h2 style='margin: 0.5rem 0 0 0; background: linear-gradient(90deg, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>RAG Assistant</h2>
            <p style='color: #6B7280; font-size: 0.85rem; margin-top: 0.25rem;'>Powered by AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Session Stats - More visual
    st.subheader("📊 Session Overview")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len(user_msgs), help="Total questions asked")
    with col2:
        st.metric("Answers", len(assistant_msgs), help="Total responses")
    
    if st.session_state.messages:
        st.divider()
        st.subheader("💾 Export Chat")
        
        export_data = prepare_export_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 JSON",
                data=export_json(export_data),
                file_name=f"rag_history_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 CSV",
                data=export_csv(export_data),
                file_name=f"rag_history_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    
    # Quick Tips
    with st.expander("💡 Tips for Better Results", expanded=False):
        st.markdown("""
        - **Be specific** — Ask detailed questions
        - **Reference docs** — Mention specific topics
        - **Follow up** — Ask clarifying questions
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.75rem; color: #9CA3AF; text-align: center;'>
            <b>Tech Stack</b><br>
            Groq • FAISS • LangChain<br>
            <span style='font-size: 0.65rem;'>v1.0.0</span>
        </div>
    """, unsafe_allow_html=True)

# Set default retrieval values (no sliders needed)
top_k = 10  # Full retrieval
confidence_threshold = 0.0  # No threshold - show confidence per query

with st.sidebar:
    st.subheader("🧪 RAG Diagnostics")

    if rag_system:
        st.metric(
            "FAISS Vectors",
            rag_system.vector_store.index.ntotal
        )

        st.metric(
            "Documents Loaded",
            len(rag_system.vector_store.documents)
        )

        st.metric(
            "Index Ready",
            "Yes" if st.session_state.get("rag_index_built") else "No"
        )


# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Header
st.markdown('<h1 class="main-header">RAG PDF Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents with AI-powered precision.</p>', unsafe_allow_html=True)

# Initialize RAG System
rag_system = get_rag_system()

if not rag_system:
    st.warning("⚠️ RAG System is initializing or encountered an error. Please check the logs.")

# Display Chat History
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # If it's an assistant message, show sources and metrics in an expander
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources & Metrics"):
                # Metrics Row
                m_col1, m_col2, m_col3 = st.columns(3)
                metrics = message.get("metrics", {})
                m_col1.metric("Confidence", f"{metrics.get('confidence', 0):.1%}")
                m_col2.metric("Sources Used", len(message.get("sources", [])))
                m_col3.metric("Time", f"{metrics.get('time', 0):.2f}s")
                
                st.divider()
                
                # Sources List
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                        <div class="source-card">
                            <div style="font-weight: 600; margin-bottom: 0.25rem;">
                                📄 {src.get('file', 'Unknown File')} (Page {src.get('page', '?')})
                            </div>
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                                {src.get('content', '')[:200]}...
                            </div>
                            <div style="font-size: 0.8rem; color: #666;">
                                Relevance: {src.get('similarity', 0):.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

if prompt := st.chat_input("💬 Ask a question about your documents..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)


    if rag_system:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Query the RAG system
                response = rag_system.query(
                    question=prompt,
                    top_k=top_k,
                    score_threshold=confidence_threshold
                )
                
                elapsed_time = time.time() - start_time
                
                answer = response.get('answer', 'I could not generate an answer.')
                sources = response.get('sources', [])
                avg_similarity = response.get('avg_similarity', 0)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources preview immediately
                if sources:
                    with st.expander("📚 Sources & Details", expanded=False):
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Confidence", f"{avg_similarity:.1%}")
                        m_col2.metric("Sources", len(sources))
                        m_col3.metric("Time", f"{elapsed_time:.2f}s")
                        
                        st.divider()
                        
                        for src in sources:
                            st.markdown(f"- **{src.get('file', 'Doc')}** (Page {src.get('page', '?')}): {src.get('similarity', 0):.1%}")

            # Add assistant message to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "metrics": {
                    "confidence": avg_similarity,
                    "time": elapsed_time
                },
                "timestamp": datetime.now().isoformat()
            })
    else:
        st.error("RAG System not ready.")