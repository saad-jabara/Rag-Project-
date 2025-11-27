#!/usr/bin/env python3
"""
Test script for RAG System
Tests basic functionality without web interface
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    elif api_key.startswith("sk-your"):
        print("❌ OPENAI_API_KEY is still a placeholder")
        print("   Please add your actual API key to the .env file")
        return False
    else:
        print("✅ OpenAI API key configured")
        return True

def test_imports():
    """Test if all required packages can be imported"""
    try:
        print("\nTesting imports...")
        import langchain
        print("✅ langchain imported")

        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("✅ langchain_openai imported")

        import chromadb
        print("✅ chromadb imported")

        import streamlit
        print("✅ streamlit imported")

        from rag_system_improved import RAGSystem
        print("✅ RAGSystem imported")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_rag_initialization():
    """Test RAG system initialization"""
    print("\nTesting RAG system initialization...")

    try:
        from rag_system_improved import RAGSystem

        # Initialize system
        print("Initializing RAG system...")
        rag = RAGSystem()

        # Test basic initialization
        print("✅ RAG system created")

        # Test if we can load sample data (this won't actually load from web in test)
        print("\n⚠️  Note: Full initialization requires internet access to load Basecamp handbook")
        print("   and will consume OpenAI API credits for embeddings.")

        return True
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG SYSTEM TEST SUITE")
    print("=" * 60)

    # Test 1: API Key
    if not test_api_key():
        print("\n⚠️  Please configure your OpenAI API key in .env file:")
        print("   1. Copy .env.example to .env if not done")
        print("   2. Add your OpenAI API key")
        print("   3. Get a key from: https://platform.openai.com/api-keys")
        return

    # Test 2: Imports
    if not test_imports():
        print("\n⚠️  Some packages are missing. Run:")
        print("   pip install -r requirements.txt")
        return

    # Test 3: RAG System
    if not test_rag_initialization():
        print("\n⚠️  RAG system initialization failed")
        return

    print("\n" + "=" * 60)
    print("✅ All basic tests passed!")
    print("=" * 60)

    print("\nYou can now run the system using one of these commands:")
    print("\n1. Web Interface (Recommended):")
    print("   streamlit run app_improved.py")

    print("\n2. Original Web Interface:")
    print("   streamlit run app.py")

    print("\n3. Command Line Interface:")
    print("   python rag_system_improved.py")

    print("\n⚠️  Note: First run will take time to:")
    print("   - Download web pages from Basecamp handbook")
    print("   - Create embeddings (uses OpenAI API)")
    print("   - Build vector database")
    print("\nSubsequent runs will be faster as embeddings are cached.")

if __name__ == "__main__":
    main()