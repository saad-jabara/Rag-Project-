"""
Improved RAG System for Basecamp Employee Handbook
Author: Saad Jabara
Based on: "Learn RAG From Scratch" tutorial
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

class RAGSystem:
    """Complete RAG system for querying Basecamp Employee Handbook"""

    def __init__(self):
        """Initialize the RAG system"""
        load_dotenv()

        # Verify API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY in your .env file")

        # URLs from Basecamp Employee Handbook
        self.basecamp_urls = [
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

        # Initialize components
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_data(self):
        """Load documents from Basecamp handbook"""
        print("Loading documents from Basecamp Employee Handbook...")
        loader = WebBaseLoader(self.basecamp_urls)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")
        return self.documents

    def split_text(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Split documents into chunks"""
        print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")

        if not self.documents:
            raise ValueError("No documents loaded. Run load_data() first.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunks)} text chunks")
        return self.chunks

    def create_embeddings_and_index(self, persist_directory: str = "./chroma_db"):
        """Create embeddings and build vector store"""
        print("Creating embeddings and building vector store...")

        if not self.chunks:
            raise ValueError("No chunks created. Run split_text() first.")

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        # Always recreate the vector store for now to ensure it works
        # In production, you'd want to check if the existing store is valid
        try:
            print(f"Creating vector store at {persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"Vector store ready with {len(self.chunks)} embedded chunks")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            # Try without persistence if there's an issue
            print("Creating in-memory vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings
            )
            print(f"In-memory vector store ready with {len(self.chunks)} embedded chunks")

        return self.vectorstore

    def setup_retrieval(self, k: int = 3):
        """Setup retrieval system"""
        print(f"Setting up retrieval (k={k})...")

        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            raise ValueError("No vector store created. Run create_embeddings_and_index() first.")

        try:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            print(f"Retriever configured to fetch top {k} relevant chunks")
        except Exception as e:
            print(f"Error setting up retriever: {e}")
            raise ValueError(f"Failed to setup retriever: {e}")

        return self.retriever

    def setup_generation(self, model: str = "gpt-3.5-turbo", temperature: float = 0):
        """Setup generation pipeline with LLM"""
        print(f"Setting up generation with {model}...")

        if not self.retriever:
            raise ValueError("No retriever created. Run setup_retrieval() first.")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )

        # Create custom prompt
        prompt_template = """Use the following pieces of context from the Basecamp Employee Handbook to answer the question.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always be specific and cite the relevant information from the context.

Context:
{context}

Question: {question}

Answer:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create the chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        self.qa_chain = {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        } | self.prompt | self.llm | StrOutputParser()

        print(f"QA chain created with {model}")
        return self.qa_chain

    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("System not initialized. Run all setup methods first.")

        # Get answer from RAG system
        answer = self.qa_chain.invoke(question)

        # Get source documents
        source_documents = self.retriever.get_relevant_documents(question)

        # Format response
        response = {
            "question": question,
            "answer": answer,
            "source_documents": source_documents
        }

        return response

    def initialize_all(self):
        """Initialize all components in sequence"""
        self.load_data()
        self.split_text()
        self.create_embeddings_and_index()
        self.setup_retrieval()
        self.setup_generation()
        print("\nRAG System fully initialized and ready!")
        return self


# Simple CLI for testing
def main():
    """Main function for CLI testing"""
    print("=" * 80)
    print("RAG SYSTEM - BASECAMP EMPLOYEE HANDBOOK")
    print("=" * 80)

    # Initialize system
    rag = RAGSystem()
    rag.initialize_all()

    print("\n" + "=" * 80)
    print("Interactive Q&A Session")
    print("Type 'quit' to exit")
    print("=" * 80)

    while True:
        print("\n")
        question = input("Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            # Get response
            result = rag.query(question)

            # Display answer
            print("\n" + "-" * 40)
            print("Answer:")
            print(result["answer"])

            # Display sources
            print("\n" + "-" * 40)
            print(f"Sources ({len(result['source_documents'])} chunks):")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n  [{i}] {doc.metadata.get('source', 'Unknown')}")
                print(f"      Preview: {doc.page_content[:150]}...")

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()