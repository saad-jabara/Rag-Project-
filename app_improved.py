"""
Improved Streamlit UI for RAG System
Author: Saad Jabara
A modern, user-friendly interface for the Basecamp Handbook RAG system
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_system_improved import RAGSystem
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

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #4A90E2 0%, #7B68EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4A90E2 0%, #7B68EE 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
    }
    .source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4A90E2;
        margin: 1rem 0;
    }
    .question-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready {
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.processing = False

# Header
st.markdown('<h1 class="main-header">🤖 RAG System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Knowledge Base for Basecamp Employee Handbook</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")

    if st.session_state.initialized:
        st.success("✅ System Ready")
        st.markdown('<span class="status-indicator status-ready"></span> **Online**', unsafe_allow_html=True)
    else:
        st.warning("⚠️ System Not Initialized")

    st.markdown("---")

    st.markdown("### 📚 About this System")
    with st.expander("Technical Details", expanded=False):
        st.markdown("""
        **Components:**
        - 🧠 **LLM**: GPT-3.5-turbo
        - 📊 **Embeddings**: text-embedding-ada-002
        - 💾 **Vector DB**: ChromaDB
        - 🔧 **Framework**: LangChain

        **Configuration:**
        - Chunk Size: 500 chars
        - Chunk Overlap: 100 chars
        - Retrieval K: 3
        - Temperature: 0
        """)

    with st.expander("Data Sources", expanded=False):
        st.markdown("""
        **Basecamp Handbook Pages:**
        - How We Work
        - Benefits and Perks
        - Work-Life Balance
        - Communication
        - Getting Started
        - DEI Initiatives
        - Internal Systems
        - And more...
        """)

    st.markdown("---")

    # API Key check
    st.markdown("### 🔑 API Configuration")
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        st.success("✅ OpenAI API Key Configured")
    else:
        st.error("❌ OpenAI API Key Missing")
        st.info("Please set your API key in the .env file")

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    **Saad Jabara**
    - 🌐 [Portfolio](https://saad-jabara.github.io)
    - 💼 [GitHub](https://github.com/saad-jabara)
    """)

# Main content area
if not st.session_state.initialized:
    # Initialization section
    st.markdown("### 🚀 Get Started")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Click the button below to initialize the RAG system. This will load documents, create embeddings, and set up the QA pipeline.")

        if st.button("🎯 Initialize RAG System", type="primary", use_container_width=True):
            # Check API key
            if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
                st.error("❌ Please configure your OpenAI API key in the .env file first!")
                st.stop()

            with st.spinner("Initializing RAG System..."):
                try:
                    # Create progress container
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Initialize RAG system
                        rag = RAGSystem()

                        # Step 1: Load data
                        status_text.text("📚 Loading documents from Basecamp Handbook...")
                        progress_bar.progress(20)
                        rag.load_data()
                        time.sleep(0.5)

                        # Step 2: Split text
                        status_text.text("✂️ Splitting text into chunks...")
                        progress_bar.progress(40)
                        rag.split_text()
                        time.sleep(0.5)

                        # Step 3: Create embeddings
                        status_text.text("🧮 Creating embeddings and indexing...")
                        progress_bar.progress(60)
                        rag.create_embeddings_and_index()
                        time.sleep(0.5)

                        # Step 4: Setup retrieval
                        status_text.text("🔍 Setting up retrieval system...")
                        progress_bar.progress(80)
                        rag.setup_retrieval()
                        time.sleep(0.5)

                        # Step 5: Setup generation
                        status_text.text("🤖 Configuring generation pipeline...")
                        progress_bar.progress(100)
                        rag.setup_generation()
                        time.sleep(0.5)

                        # Store in session state
                        st.session_state.rag_system = rag
                        st.session_state.initialized = True

                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()

                    st.balloons()
                    st.success("✅ RAG System initialized successfully!")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
                    st.info("Please check your OpenAI API key and internet connection.")

else:
    # Query Interface
    st.markdown("### 💬 Ask Your Question")

    # Create tabs for different interaction modes
    tab1, tab2 = st.tabs(["🔍 Query", "📜 History"])

    with tab1:
        # Question input
        question = st.text_area(
            "Enter your question about the Basecamp Employee Handbook:",
            placeholder="Example: What benefits does Basecamp offer employees?",
            height=100,
            key="question_input"
        )

        col1, col2 = st.columns([4, 1])
        with col2:
            submit_button = st.button("🔍 Search", type="primary", use_container_width=True, disabled=st.session_state.processing)

        # Example questions
        st.markdown("#### 💡 Quick Examples")
        example_cols = st.columns(3)
        example_questions = [
            "What benefits does Basecamp offer?",
            "How does Basecamp handle remote work?",
            "What is the vacation policy?",
            "How does internal communication work?",
            "What are Basecamp's core values?",
            "How does Basecamp support work-life balance?"
        ]

        for i, example in enumerate(example_questions):
            with example_cols[i % 3]:
                if st.button(f"💡 {example[:25]}...", key=f"ex_{i}", use_container_width=True):
                    question = example
                    submit_button = True

        # Process query
        if submit_button and question:
            st.session_state.processing = True

            with st.spinner("🔄 Processing your question..."):
                try:
                    # Query the RAG system
                    start_time = time.time()
                    result = st.session_state.rag_system.query(question)
                    query_time = time.time() - start_time

                    # Add to chat history
                    st.session_state.chat_history.insert(0, {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["source_documents"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query_time": query_time
                    })

                    # Display result
                    st.success(f"✅ Answer generated in {query_time:.2f} seconds")

                    # Answer section
                    st.markdown("#### 💡 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                    # Sources section
                    with st.expander(f"📚 View Source Documents ({len(result['source_documents'])} chunks)", expanded=False):
                        for i, doc in enumerate(result['source_documents'], 1):
                            source_url = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**Source {i}:** [{source_url}]({source_url})")
                            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    st.session_state.processing = False

    with tab2:
        # Chat History
        if st.session_state.chat_history:
            st.markdown("#### 📜 Conversation History")

            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

            # Display history
            for i, item in enumerate(st.session_state.chat_history):
                with st.container():
                    col1, col2 = st.columns([10, 2])
                    with col1:
                        st.markdown(f'<div class="question-box">❓ {item["question"]}</div>', unsafe_allow_html=True)
                    with col2:
                        st.caption(f"🕐 {item['timestamp']}")
                        st.caption(f"⚡ {item['query_time']:.2f}s")

                    st.markdown(item["answer"])

                    with st.expander(f"Sources ({len(item['sources'])} chunks)"):
                        for j, doc in enumerate(item['sources'], 1):
                            st.caption(f"Source {j}: {doc.metadata.get('source', 'Unknown')}")
                            st.text(doc.page_content[:200] + "...")

                    st.markdown("---")
        else:
            st.info("No questions asked yet. Start by asking a question in the Query tab!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p>Built with ❤️ by Saad Jabara |
    <a href='https://github.com/saad-jabara/RAG-from-Scratch' style='color: #4A90E2;'>GitHub</a> |
    <a href='https://saad-jabara.github.io' style='color: #4A90E2;'>Portfolio</a>
    </p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
        Powered by LangChain, ChromaDB, and OpenAI GPT-3.5
    </p>
</div>
""", unsafe_allow_html=True)