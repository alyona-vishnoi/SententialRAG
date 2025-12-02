# src/basic_rag.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment
load_dotenv()

class BasicRAG:
    """A basic RAG system that WILL hallucinate (intentionally!)"""
    
    def __init__(self):
        print("🔧 Initializing Basic RAG...")
        
        # Initialize LLM (Gemini - FREE!)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7  # Higher = more creative = more hallucinations!
        )
        
        # Initialize embeddings (FREE local model!)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = None
        
        print("RAG initialized!")
    
    def load_documents(self):
        """Load processed papers"""
        print("\nLoading documents...")
        
        data_path = Path("data/papers/processed_papers.json")
        with open(data_path, 'r') as f:
            papers = json.load(f)
        
        # Convert to LangChain Documents
        documents = []
        for paper in papers:
            doc = Document(
                page_content=paper['text'],
                metadata={
                    'title': paper['title'],
                    'authors': ', '.join(paper['authors'][:3]),  # First 3 authors
                    'published': paper['published'],
                    'arxiv_id': paper['arxiv_id']
                }
            )
            documents.append(doc)
        
        print(f"Loaded {len(documents)} papers")
        return documents
    
    def create_vector_store(self, documents):
        """Chunk documents and create vector store"""
        print("\n🔪 Splitting documents into chunks...")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 1000 characters per chunk
            chunk_overlap=200,  # 200 char overlap
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        print("\nCreating embeddings and vector store...")
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="data/chroma_db"
        )
        
        print("Vector store created!")
    
    def query(self, question, num_results=3):
        """Query the RAG system"""
        
        if self.vector_store is None:
            raise Exception("Vector store not initialized! Call create_vector_store() first")
        
        print(f"\nQuestion: {question}")
        print("=" * 60)
        
        # Step 1: Retrieve relevant chunks
        print("\nSearching for relevant documents...")
        results = self.vector_store.similarity_search(question, k=num_results)
        
        print(f"Found {len(results)} relevant chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n   [{i}] {doc.metadata['title'][:50]}...")
            print(f"       {doc.page_content[:100]}...")
        
        # Step 2: Build context
        context = "\n\n".join([
            f"Document {i} ({doc.metadata['title']}):\n{doc.page_content}"
            for i, doc in enumerate(results, 1)
        ])
        
        # Step 3: Generate answer (THIS IS WHERE HALLUCINATIONS HAPPEN!)
        print("\nGenerating answer...")
        
        prompt = f"""Answer the question based on the context below.

        Context:
        {context}

        Question: {question}

        Answer (be specific and cite sources):"""
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(answer)
        print("=" * 60)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [doc.metadata for doc in results]
        }

def main():
    """Demo the basic RAG system"""
    
    print("BASIC RAG DEMO - Let's Make It Hallucinate!")
    print("=" * 60)
    
    # Initialize RAG
    rag = BasicRAG()
    
    # Load documents
    documents = rag.load_documents()
    
    # Create vector store (only do this once!)
    if not Path("data/chroma_db").exists():
        rag.create_vector_store(documents)
    else:
        print("\nVector store already exists, loading...")
        rag.vector_store = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=rag.embeddings
        )
    
    print("\n" + "=" * 60)
    print("Now let's ask some questions that might cause hallucinations!")
    print("=" * 60)
    
    # Test questions (some will cause hallucinations!)
    test_questions = [
        "How many parameters does GPT-3 have?",
        "What is the architecture of BERT?",
        "When was the transformer architecture introduced?",
        "What are the main contributions of the Vision Transformer paper?",
    ]
    
    results = []
    for question in test_questions:
        result = rag.query(question)
        results.append(result)
        
        input("\nPress Enter to continue to next question...")
    
    # Save results
    output_path = Path("data/rag_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nDemo complete!")

if __name__ == "__main__":
    main()