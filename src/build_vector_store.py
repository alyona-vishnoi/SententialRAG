"""
Build vector store with Famous + Recent ML papers
"""

import json
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.better_chunking import SmartChunker

def build_vector_store():
    """Build vector store with all papers"""
    
    print("Building vector store with ALL papers...")
    print("="*60)
    
    # Load processed papers
    processed_path = Path("data/papers/all_papers_processed.json")
    
    with open(processed_path, 'r') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers")
    
    # Convert to LangChain Documents
    documents = []
    for paper in papers:
        doc = Document(
            page_content=paper['text'],
            metadata={
                'title': paper['title'],
                'arxiv_id': paper['arxiv_id'],
                'category': paper['category'],
                'authors': ', '.join(paper['authors'][:3]),  # First 3 authors
                'published': paper['published'],
                'num_pages': paper['num_pages']
            }
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} documents")
    
    # Chunking
    chunker = SmartChunker()
    print("\nSplitting into chunks...")
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks")
    
    # Show chunk distribution
    famous_chunks = sum(1 for c in all_chunks if c.metadata.get('category') == 'famous')
    recent_chunks = sum(1 for c in all_chunks if c.metadata.get('category') == 'recent')
    important_chunks = sum(1 for c in all_chunks if c.metadata.get('is_important', False))
    print(f"Famous paper chunks: {famous_chunks}")
    print(f"Recent paper chunks: {recent_chunks}")
    print(f"Important chunks: {important_chunks}")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="data/chroma_db"
    )
    
    print("Vector store created!")
    
    # Test it with multiple queries
    print("\nTesting vector store...")
    test_queries = [
        ("GPT-3 parameters", "Should find GPT-3 paper"),
        ("BERT architecture", "Should find BERT paper"),
        ("transformer attention", "Should find Transformer paper"),
        ("recent advances", "Should find recent papers"),
    ]
    
    for query, expected in test_queries:
        print(f"\n   Query: '{query}' ({expected})")
        results = vector_store.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get('title', 'Unknown')[:45]
            category = doc.metadata.get('category', '?')
            print(f"      [{i}] [{category.upper()}] {title}...")
    
    print("\n"+"="*60)
    print("Vector store built successfully!")
    print("="*60)

if __name__ == "__main__":
    build_vector_store()