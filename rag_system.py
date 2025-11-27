"""
RAG from Scratch - Basecamp Employee Handbook RAG System
Based on: "Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer"
YouTube: https://www.youtube.com/watch?v=sVcwVQRHIc8

This script implements a complete Retrieval-Augmented Generation (RAG) system using:
- LangChain for orchestration
- ChromaDB for vector storage
- OpenAI API for embeddings and generation
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

print("=" * 80)
print("RAG FROM SCRATCH - BASECAMP EMPLOYEE HANDBOOK")
print("=" * 80)

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================
print("\n[STEP 1] Loading documents from Basecamp Employee Handbook...")

# URLs from Basecamp Employee Handbook
basecamp_urls = [
    "https://basecamp.com/handbook",
    "https://basecamp.com/handbook/how-we-work",
    "https://basecamp.com/handbook/benefits-and-perks",
    "https://basecamp.com/handbook/work-life-balance",
    "https://basecamp.com/handbook/titles-for-support",
    "https://basecamp.com/handbook/getting-started",
    "https://basecamp.com/handbook/communication",
    "https://basecamp.com/handbook/our-internal-systems",
    "https://basecamp.com/handbook/pricing-and-profit",
    "https://basecamp.com/handbook/dei"
]

# Load documents using WebBaseLoader
loader = WebBaseLoader(basecamp_urls)
documents = loader.load()

print(f"✓ Loaded {len(documents)} documents from Basecamp handbook")
print(f"\nDocument preview (first 200 characters):")
print(f"{documents[0].page_content[:200]}...")

# =============================================================================
# STEP 2: TEXT SPLITTING
# =============================================================================
print("\n[STEP 2] Splitting documents into chunks...")

# Initialize text splitter with chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"✓ Created {len(chunks)} text chunks")
print(f"\nChunk preview (first chunk):")
print(f"{chunks[0].page_content[:300]}...")

# =============================================================================
# STEP 3: EMBEDDING & INDEXING
# =============================================================================
print("\n[STEP 3] Creating embeddings and building vector store...")

# Initialize OpenAI embeddings (text-embedding-ada-002, 1536 dimensions)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# Create Chroma vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✓ Embeddings created and stored in ChromaDB")
print(f"✓ Vector store contains {len(chunks)} embedded chunks")

# =============================================================================
# STEP 4: RETRIEVAL
# =============================================================================
print("\n[STEP 4] Setting up retrieval...")

# Create retriever that returns top 3 relevant chunks
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print(f"✓ Retriever configured to fetch top 3 relevant chunks")

# =============================================================================
# STEP 5: GENERATION
# =============================================================================
print("\n[STEP 5] Setting up generation with GPT-3.5-turbo...")

# Initialize ChatOpenAI with GPT-3.5-turbo
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Create custom prompt template
prompt_template = """Use the following pieces of context from the Basecamp Employee Handbook to answer the question.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print(f"✓ QA chain created with GPT-3.5-turbo")

# =============================================================================
# STEP 6: EXAMPLE QUERIES
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE QUERIES")
print("=" * 80)

# Define example questions
example_questions = [
    "What benefits does Basecamp offer employees?",
    "How does Basecamp support work-life balance?",
    "What is Basecamp's approach to internal communication?"
]

# Process each question
for i, question in enumerate(example_questions, 1):
    print(f"\n{'='*80}")
    print(f"QUERY {i}")
    print(f"{'='*80}")
    print(f"\nQuestion: {question}")

    # Get answer from RAG system
    result = qa_chain.invoke({"query": question})

    print(f"\nAnswer:\n{result['result']}")

    print(f"\nSource Documents ({len(result['source_documents'])} chunks used):")
    for j, doc in enumerate(result['source_documents'], 1):
        print(f"\n  Chunk {j}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Content preview: {doc.page_content[:200]}...")

print("\n" + "=" * 80)
print("RAG SYSTEM DEMO COMPLETE")
print("=" * 80)
print("\nYou can now use this system to ask questions about the Basecamp Employee Handbook!")
print("The system retrieves relevant context and generates accurate answers using GPT-3.5-turbo.")
