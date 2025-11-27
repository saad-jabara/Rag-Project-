"""
Streamlit UI for RAG System
Author: Saad Jabara
A web interface for the Basecamp Handbook RAG system
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System - Basecamp Handbook",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4A90E2;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.chat_history = []

# Header
st.markdown('<h1 class="main-header">🤖 RAG System: Basecamp Handbook</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about the Basecamp Employee Handbook using AI-powered retrieval</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📚 About this System")
    st.info("""
    This RAG (Retrieval-Augmented Generation) system:
    - **Data Source**: Basecamp Employee Handbook
    - **Embedding Model**: OpenAI text-embedding-ada-002
    - **LLM**: GPT-3.5-turbo
    - **Vector Store**: ChromaDB
    - **Framework**: LangChain
    """)

    st.markdown("## 🎓 Tutorial Credit")
    st.markdown("""
    Based on: [Learn RAG From Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8)
    - YouTube tutorial by LangChain Engineer
    """)

    st.markdown("## 👨‍💻 Developer")
    st.markdown("""
    **Saad Jabara**
    - AI Engineer | RAG Practitioner
    - [GitHub](https://github.com/saad-jabara)
    - [Portfolio](https://saad-jabara.github.io)
    """)

    # System metrics
    if st.session_state.initialized:
        st.markdown("## 📊 System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", "10")
            st.metric("Chunks", "~500")
        with col2:
            st.metric("Embedding Dim", "1536")
            st.metric("Retrieval K", "3")

# Main content area
main_container = st.container()

with main_container:
    # Initialize System Button
    if not st.session_state.initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize RAG System", type="primary"):
                with st.spinner("Initializing RAG System... This may take a moment."):
                    try:
                        # Initialize RAG system
                        rag = RAGSystem()

                        # Progress indicators
                        progress = st.progress(0)
                        status = st.empty()

                        # Load data
                        status.text("📚 Loading documents from Basecamp Handbook...")
                        progress.progress(20)
                        rag.load_data()

                        # Split text
                        status.text("✂️ Splitting text into chunks...")
                        progress.progress(40)
                        rag.split_text()

                        # Create embeddings
                        status.text("🧮 Creating embeddings and indexing...")
                        progress.progress(60)
                        rag.create_embeddings_and_index()

                        # Setup retrieval
                        status.text("🔍 Setting up retrieval system...")
                        progress.progress(80)
                        rag.setup_retrieval()

                        # Setup generation
                        status.text("🤖 Configuring generation pipeline...")
                        progress.progress(100)
                        rag.setup_generation()

                        # Store in session state
                        st.session_state.rag_system = rag
                        st.session_state.initialized = True

                        # Clear progress indicators
                        progress.empty()
                        status.empty()

                        st.success("✅ RAG System initialized successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to initialize: {str(e)}")
                        st.info("Please ensure your OpenAI API key is set in the .env file")

    else:
        # Query Interface
        st.markdown("### 💬 Ask a Question")

        # Example questions
        example_questions = [
            "What benefits does Basecamp offer employees?",
            "How does Basecamp support work-life balance?",
            "What is Basecamp's approach to internal communication?",
            "Tell me about Basecamp's vacation policy",
            "How does Basecamp handle remote work?",
            "What are Basecamp's core values?"
        ]

        # Question input
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Your question:",
                placeholder="Ask anything about the Basecamp Employee Handbook...",
                label_visibility="collapsed"
            )

        with col2:
            if st.button("🔍 Search", type="primary", disabled=not question):
                process_query = True
            else:
                process_query = False

        # Example questions buttons
        st.markdown("**Quick examples:**")
        cols = st.columns(3)
        for i, example in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(f"📝 {example[:30]}...", key=f"example_{i}"):
                    question = example
                    process_query = True

        # Process query
        if process_query and question:
            with st.spinner("🔄 Processing your question..."):
                try:
                    # Query the RAG system
                    result = st.session_state.rag_system.query(question)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["source_documents"]
                    })

                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")

            for i, item in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # Question
                    st.markdown(f"**❓ Question {len(st.session_state.chat_history) - i}:**")
                    st.markdown(f"> {item['question']}")

                    # Answer
                    st.markdown("**💡 Answer:**")
                    st.markdown(item['answer'])

                    # Sources
                    with st.expander(f"📚 View Source Documents ({len(item['sources'])} chunks)"):
                        for j, doc in enumerate(item['sources'], 1):
                            st.markdown(f"**Source {j}:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")

                    st.markdown("---")

        # Clear history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation History"):
                st.session_state.chat_history = []
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ by Saad Jabara |
    <a href='https://github.com/saad-jabara/RAG-from-Scratch'>GitHub</a> |
    <a href='https://saad-jabara.github.io'>Portfolio</a>
    </p>
    <p style='font-size: 0.9rem;'>Powered by LangChain, ChromaDB, and OpenAI</p>
</div>
""", unsafe_allow_html=True)