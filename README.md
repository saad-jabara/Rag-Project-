# RAG from Scratch 🤖

A complete Retrieval-Augmented Generation (RAG) system built from scratch using LangChain, ChromaDB, and OpenAI's API. This project demonstrates how to build an intelligent question-answering system trained on the Basecamp Employee Handbook.

## 📺 Tutorial Credit

This project is based on the excellent YouTube tutorial:
**"Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer"**
- 🔗 [Watch on YouTube](https://www.youtube.com/watch?v=sVcwVQRHIc8)
- Created by LangChain Engineer

## 🎯 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using an LLM to generate accurate answers based on retrieved context

This approach allows AI models to answer questions accurately using specific, up-to-date information from custom documents, rather than relying solely on their training data.

## 🏗️ Project Architecture

```
User Question
      ↓
[1. Embed Question]
      ↓
[2. Retrieve Relevant Chunks] ← Vector Store (ChromaDB)
      ↓
[3. Generate Answer] ← LLM (GPT-3.5-turbo)
      ↓
Final Answer + Sources
```

## 🛠️ Tech Stack

- **Python 3.8+**
- **LangChain** - Framework for building LLM applications
- **ChromaDB** - Vector database for storing embeddings
- **OpenAI API** - Embeddings (text-embedding-ada-002) and LLM (gpt-3.5-turbo)
- **WebBaseLoader** - Loading web content
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

## 📚 Knowledge Base

This RAG system is trained on the **Basecamp Employee Handbook**, including:
- How We Work
- Benefits and Perks
- Work-Life Balance
- Internal Communication
- Getting Started Guide
- Diversity, Equity, and Inclusion (DEI)
- Pricing and Profit
- Support Titles
- Internal Systems

## 🚀 Quick Start

### Prerequisites

1. Python 3.8 or higher
2. OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/saadthespecialist/RAG-from-Scratch.git
cd RAG-from-Scratch
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

5. **Run the RAG system**

**Option 1: Command Line Interface**
```bash
python rag_system.py
```

**Option 2: Web Interface (Streamlit)**
```bash
streamlit run app.py
```
The web UI will open at http://localhost:8501

## 💡 How It Works

### Step 1: Data Loading
```python
# Load 10 pages from Basecamp Employee Handbook
loader = WebBaseLoader(basecamp_urls)
documents = loader.load()
```

### Step 2: Text Splitting
```python
# Split documents into 500-character chunks with 100-character overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
```

### Step 3: Embedding & Indexing
```python
# Create embeddings using OpenAI's text-embedding-ada-002 (1536 dimensions)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(chunks, embeddings)
```

### Step 4: Retrieval
```python
# Retrieve top 3 most relevant chunks for each query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### Step 5: Generation
```python
# Generate answers using GPT-3.5-turbo with retrieved context
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

## 📊 Example Queries & Outputs

### Query 1: Benefits
**Q:** "What benefits does Basecamp offer employees?"

**A:** Basecamp offers comprehensive benefits including health insurance, retirement plans, paid time off, professional development opportunities, and flexible work arrangements.

**Sources:** 3 relevant chunks from the Benefits and Perks page

### Query 2: Work-Life Balance
**Q:** "How does Basecamp support work-life balance?"

**A:** Basecamp supports work-life balance through flexible schedules, remote work options, reasonable working hours, and a strong emphasis on avoiding burnout.

**Sources:** 3 relevant chunks from the Work-Life Balance page

### Query 3: Communication
**Q:** "What is Basecamp's approach to internal communication?"

**A:** Basecamp emphasizes asynchronous communication, written documentation, and transparency. They use their own Basecamp product for internal collaboration.

**Sources:** 3 relevant chunks from the Communication page

## 🔑 Key Features

✅ **Accurate Retrieval** - Finds the most relevant information using semantic search
✅ **Source Attribution** - Shows which documents were used to generate answers
✅ **Context-Aware** - Answers are grounded in actual handbook content
✅ **Efficient Chunking** - Optimized chunk size (500 chars) with overlap (100 chars)
✅ **Production-Ready** - Uses industry-standard tools (LangChain, ChromaDB, OpenAI)

## 📁 Project Structure

```
RAG-from-Scratch/
├── rag_system.py          # Main RAG implementation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
└── chroma_db/            # Vector database (created on first run)
```

## 🧠 What I Learned

This project covers all major RAG concepts from the tutorial:

✅ **Overview** - Understanding RAG architecture
✅ **Indexing** - Document loading and vector storage
✅ **Retrieval** - Semantic search and relevance ranking
✅ **Generation** - LLM-based answer generation
✅ **Query Translation** - Multi-Query, RAG Fusion, Decomposition
✅ **Step Back & HyDE** - Advanced query techniques
✅ **Routing** - Intelligent query routing
✅ **Query Construction** - Structured queries
✅ **Multi-Representation Indexing** - RAPTOR, ColBERT
✅ **CRAG** - Corrective RAG
✅ **Adaptive RAG** - Dynamic retrieval strategies

## 🤝 Contributing

Feel free to fork this repository and experiment with:
- Different document sources
- Alternative embedding models
- Custom prompt templates
- Advanced retrieval strategies

## 📝 License

MIT License - Feel free to use this code for learning and building your own RAG systems!

## 👨‍💻 Author

**Saad Jabara**
AI Engineer | Python Developer | LangChain Practitioner

- 🌐 [Portfolio](https://saadthespecialist.github.io)
- 💼 [LinkedIn](https://linkedin.com/in/saadthespecialist)
- 🐙 [GitHub](https://github.com/saadthespecialist)

---

⭐ If you found this helpful, please star the repository!

Built with ❤️ following the "Learn RAG From Scratch" tutorial
