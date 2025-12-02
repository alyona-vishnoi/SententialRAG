"""
Shared state for the multi-agent RAG system
"""

from typing import TypedDict, Annotated, List
from langchain.schema import Document

class RAGState(TypedDict):
    """
    State that flows through the agent graph
    Each agent reads from and writes to this state
    """
    
    # Input
    query: str  # Original user question
    
    # Query Analysis
    query_type: str
    key_entities: List[str]
    optimized_query: str
    requires_multi_doc: bool
    
    # Retrieval
    retrieved_docs: List[Document]
    retrieval_attempts: int
    retrieval_relevance: float
    retrieval_coverage: float
    
    # Answer Generation
    answer: str
    num_citations: int
    sources_used: List[str]
    
    # Fact Checking
    total_claims: int
    verified_claims: int
    failed_claims: int
    fact_check_confidence: float
    has_fact_issues: bool
    fact_issues_desc: str
    
    # Quality Scoring
    overall_score: float
    factual_accuracy: float
    citation_quality: float
    relevance: float
    completeness: float
    recommendation: str  # PASS, FLAG, REGENERATE
    feedback: str
    
    # Control flow
    should_retry_retrieval: bool
    should_regenerate: bool
    final_status: str  # SUCCESS, FAILED, NEEDS_HUMAN