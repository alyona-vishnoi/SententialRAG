"""
Hybrid retriever combining vector search and BM25 keyword search
"""

from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Combines vector similarity search with BM25 keyword search
    """
    
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.bm25 = None
        self.all_docs = None
        
        # Initialize BM25
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """
        Initialize BM25 with all documents from vector store
        """
        print("Initializing BM25 index...")
        
        # Get all documents from vector store
        # This is a workaround since Chroma doesn't expose all docs easily
        # We'll do a broad search to get many docs
        self.all_docs = self.vector_store.similarity_search("machine learning", k=1000)
        
        if not self.all_docs:
            print("No documents found in vector store")
            return
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.page_content.lower().split() for doc in self.all_docs]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"BM25 initialized with {len(self.all_docs)} documents")
    
    def retrieve(self, query: str, k: int = 5, 
                 vector_weight: float = 0.7, 
                 bm25_weight: float = 0.3) -> List[Document]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            k: Number of documents to return
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            
        Returns:
            List of documents, ranked by hybrid score
        """
        
        # If BM25 not available, fall back to vector only
        if self.bm25 is None:
            print("Using vector search only (BM25 not available)")
            return self.vector_store.similarity_search(query, k=k)
        
        print(f"Hybrid search (vector: {vector_weight:.0%}, BM25: {bm25_weight:.0%})")
        
        # 1. Vector search
        vector_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # 2. BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_results = [(self.all_docs[i], bm25_scores[i]) for i in top_bm25_indices]
        
        # 3. Normalize scores
        vector_scores_norm = self._normalize_scores([score for _, score in vector_results])
        bm25_scores_norm = self._normalize_scores([score for _, score in bm25_results])
        
        # 4. Create score dictionary
        doc_scores = {}
        
        # Add vector scores
        for i, (doc, score) in enumerate(vector_results):
            doc_id = doc.page_content[:100]  # Use text snippet as ID
            doc_scores[doc_id] = {
                'doc': doc,
                'vector_score': vector_scores_norm[i],
                'bm25_score': 0.0
            }
        
        # Add BM25 scores
        for i, (doc, score) in enumerate(bm25_results):
            doc_id = doc.page_content[:100]
            if doc_id in doc_scores:
                doc_scores[doc_id]['bm25_score'] = bm25_scores_norm[i]
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'bm25_score': bm25_scores_norm[i]
                }
        
        # 5. Compute hybrid scores
        for doc_id in doc_scores:
            doc_scores[doc_id]['hybrid_score'] = (
                vector_weight * doc_scores[doc_id]['vector_score'] +
                bm25_weight * doc_scores[doc_id]['bm25_score']
            )
        
        # 6. Sort by hybrid score
        ranked_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        # 7. Return top k documents
        top_docs = [item['doc'] for item in ranked_docs[:k]]
        
        print(f"Retrieved {len(top_docs)} documents")
        
        # Debug: Show score breakdown for top result
        if ranked_docs:
            top = ranked_docs[0]
            print(f"Top result: vector={top['vector_score']:.3f}, "
                  f"bm25={top['bm25_score']:.3f}, "
                  f"hybrid={top['hybrid_score']:.3f}")
        
        return top_docs
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


# Quick test
if __name__ == "__main__":
    print("HybridRetriever class defined successfully!")
    print("\nTo use:")
    print("  retriever = HybridRetriever(vector_store)")
    print("  results = retriever.retrieve('GPT-3 parameters', k=5)")