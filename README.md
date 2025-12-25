📚 RAG PDF Q&A System
A Retrieval-Augmented Generation (RAG) system for intelligent question-answering over PDF documents using AI embeddings, semantic search, and large language models.

✨ Features
🎯 Core Features
✅ Multi-PDF Support - Process multiple PDF documents

✅ Semantic Search - Find relevant sections using embeddings

✅ AI-Powered Answers - Generate contextual responses using Groq LLM

✅ Source Attribution - Citations with file name, page number, relevance score

✅ Chat History - Auto-save all conversations with metadata

✅ Export Options - Download as JSON, CSV, or Markdown

🎨 UI/UX Features
✅ Clean Minimalist Design - Professional, distraction-free interface

✅ Two-Column Layout - Information sidebar + main content area

✅ Responsive Design - Works on desktop, tablet, mobile

✅ Tab-Based Interface - Answer, Sources, and Statistics views

✅ Real-Time Metrics - Response time, confidence scores, source count

🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                         │
│                   (app.py / app_clean_simple.py)               │
├─────────────────────────────────────────────────────────────────┤
│                     RAG SYSTEM (src/)                           │
│  ┌──────────────┬─────────────┬──────────┬───────────────────┐ │
│  │ embeddings.py│ vector_store│retriever │ llm_client.py    │ │
│  │ (SentenceB.) │ (FAISS)     │ (search) │ (Groq API)       │ │
│  └──────────────┴─────────────┴──────────┴───────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                  EXTERNAL SERVICES                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  PDF Documents  │  │   FAISS      │  │  Groq LLM API  │   │
│  │  (Local)        │  │  (Vector DB) │  │  (Remote)      │   │
│  └─────────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

✨Performance
-->Query embedding: ~500ms
-->Vector search: ~100ms
-->LLM response: ~1.5-2s
-->=Total: 2-2.5 seconds

🎇Future Enhancements
    O User authentication

    O Document versioning

    O RAG evaluation dashboard (RAGAS)

    O Hybrid search (BM25 + semantic)

    O Multi-language support

🪁Contributing
Contributions welcome! Please submit PRs or issues.

📖License
MIT

✍️Author
RAM K - AI/ML Student

Built with ❤️ using LangChain, FAISS, and Groq