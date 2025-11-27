# 🚀 Getting Started with RAG System

## Quick Start (Easiest Method)

### Step 1: Configure Your OpenAI API Key
1. Open the `.env` file in a text editor
2. Replace `sk-your-actual-openai-api-key-here` with your actual OpenAI API key
3. Don't have a key? Get one at: https://platform.openai.com/api-keys

### Step 2: Run the System
```bash
./start.sh
```

This will:
- Set up the Python environment automatically
- Install all required dependencies
- Check your API key configuration
- Present a menu to choose how to run the system

## Available Interfaces

### 1. **Improved Web Interface** (Recommended)
- Modern, user-friendly design
- Real-time progress indicators
- Conversation history
- Query time tracking
- Source document viewer

### 2. **Original Web Interface**
- Classic Streamlit design
- Basic Q&A functionality
- Source attribution

### 3. **Command Line Interface**
- Terminal-based interaction
- Direct Q&A mode
- Good for testing and debugging

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure .env file with your OpenAI API key

# 5. Run the improved web interface
streamlit run app_improved.py
```

## What to Expect

### First Run
- The system will download pages from the Basecamp Employee Handbook
- It will create embeddings using OpenAI's API (this consumes API credits)
- A vector database will be built and cached locally
- This process takes 1-2 minutes

### Subsequent Runs
- The system will load from the cached vector database
- Startup will be much faster (< 10 seconds)

## Example Questions to Try

Once the system is running, try these questions:

1. "What benefits does Basecamp offer employees?"
2. "How does Basecamp support work-life balance?"
3. "What is Basecamp's vacation policy?"
4. "How does internal communication work at Basecamp?"
5. "What are Basecamp's core values?"
6. "How does Basecamp handle remote work?"

## Troubleshooting

### "API Key not configured" Error
- Make sure you've added your actual OpenAI API key to the `.env` file
- The key should start with `sk-` and be a valid OpenAI API key

### "Module not found" Error
- Make sure you're using the virtual environment
- Run: `source venv/bin/activate` before starting

### "Connection error" when loading documents
- Check your internet connection
- The system needs to download pages from basecamp.com

### High API Usage
- The first run will use API credits for creating embeddings
- Subsequent runs only use credits for generating answers (minimal usage)

## Project Structure

```
RAG-from-Scratch/
├── .env                    # Your API key (configure this!)
├── start.sh               # Easy startup script
├── app_improved.py        # Enhanced web interface
├── app.py                 # Original web interface
├── rag_system_improved.py # Core RAG implementation
├── rag_system.py         # Original implementation
├── test_rag.py           # Test suite
├── requirements.txt      # Python dependencies
├── chroma_db/           # Vector database (created on first run)
└── venv/               # Python virtual environment
```

## Cost Estimates

- **Initial Setup**: ~$0.10-0.20 (for creating embeddings of ~500 text chunks)
- **Per Query**: ~$0.001-0.002 (for generating answers)

## Need Help?

If you encounter issues:
1. Run the test suite: `python test_rag.py`
2. Check the terminal output for error messages
3. Ensure your OpenAI API key is valid and has credits

---

**Author**: Saad Jabara
**Based on**: "Learn RAG From Scratch" YouTube Tutorial
**Technologies**: LangChain, ChromaDB, OpenAI API, Streamlit