#!/bin/bash

# RAG System Startup Script
# Author: Saad Jabara

echo "============================================"
echo "    RAG System - Basecamp Handbook"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install/update requirements
echo "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies ready"

# Check API key
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "✅ .env file created"
fi

# Extract API key from .env
API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)

if [[ $API_KEY == *"sk-your"* ]] || [ -z "$API_KEY" ]; then
    echo ""
    echo "❌ OpenAI API Key not configured!"
    echo ""
    echo "Please add your OpenAI API key to the .env file:"
    echo "  1. Open .env file in a text editor"
    echo "  2. Replace 'sk-your-actual-openai-api-key-here' with your API key"
    echo "  3. Get a key from: https://platform.openai.com/api-keys"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ OpenAI API key configured"
echo ""

# Menu
echo "Select an option:"
echo "  1) Run Improved Web Interface (Recommended)"
echo "  2) Run Original Web Interface"
echo "  3) Run Command Line Interface"
echo "  4) Run Test Suite"
echo "  5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Starting Improved Web Interface..."
        echo "The app will open in your browser at http://localhost:8501"
        echo ""
        streamlit run app_improved.py
        ;;
    2)
        echo ""
        echo "Starting Original Web Interface..."
        echo "The app will open in your browser at http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    3)
        echo ""
        echo "Starting Command Line Interface..."
        echo ""
        python rag_system_improved.py
        ;;
    4)
        echo ""
        python test_rag.py
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac