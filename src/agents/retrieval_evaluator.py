"""
Agent 2
Retrieval Evaluator
Checks if retrieved documents are relevant and sufficient
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class RetrievalEvaluation(BaseModel):
    """Evaluation of retrieval quality"""
    is_sufficient: bool = Field(description="Are results good enough?")
    relevance_score: float = Field(description="Relevance score 0-1")
    coverage_score: float = Field(description="Coverage score 0-1")
    should_retry: bool = Field(description="Should we retry with different query?")
    retry_suggestion: str = Field(description="Suggested query if retry needed")
    reasoning: str = Field(description="Explanation of evaluation")

class RetrievalEvaluatorAgent:
    """Agent that evaluates retrieval quality"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
    
    def evaluate(self, query: str, retrieved_docs: list, num_expected: int = 3) -> RetrievalEvaluation:
        """Evaluate if retrieved documents are sufficient"""
        
        print(f"\n[Retrieval Evaluator] Checking {len(retrieved_docs)} documents...")
        
        # QUICK CHECK: Do any docs directly mention key terms?
        # Prevents overly harsh LLM evaluation
        from agents.query_analyzer import QueryAnalyzerAgent
        analyzer = QueryAnalyzerAgent()
        analysis = analyzer.analyze(query)
        key_entities = analysis.key_entities
        
        # Count how many docs mention key entities
        docs_with_entities = 0
        for doc in retrieved_docs:
            doc_text_lower = doc.page_content.lower()
            if any(entity.lower() in doc_text_lower for entity in key_entities):
                docs_with_entities += 1
        
        entity_coverage = docs_with_entities / len(retrieved_docs) if retrieved_docs else 0
        
        # If most docs mention key entities, baseline score is decent
        baseline_relevance = entity_coverage * 0.6  # Base 60% if entities found
        
        # Build document summaries for LLM
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.metadata.get('title', 'Unknown')
            snippet = doc.page_content[:300].replace('\n', ' ')
            doc_summaries.append(f"Doc {i} ({title}):\n{snippet}...")
        
        docs_text = '\n\n'.join(doc_summaries)
        
        prompt = f"""Evaluate if these documents can help answer the query.
            Query: {query}

            Documents:
            {docs_text}

            Rate these documents:
            1. RELEVANCE (0-1): Do documents discuss the topic? Even indirect mentions count!
            2. COVERAGE (0-1): Is there enough information to attempt an answer? Partial info gets 0.5+

            Be GENEROUS - if documents mention relevant concepts, give credit.
            Examples:
            - "GPT-3 175B" mentioned → Relevance 0.8, Coverage 0.7
            - Documents about transformers when asked about BERT → Relevance 0.6
            - Completely unrelated → Relevance 0.0

            RESPOND IN THIS FORMAT:
            RELEVANCE: [0.0-1.0]
            COVERAGE: [0.0-1.0]
            SUFFICIENT: [YES or NO]
            REASONING: [one sentence]
        """
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Parse
        import re
        
        relevance_match = re.search(r'RELEVANCE:\s*([0-9.]+)', content, re.IGNORECASE)
        coverage_match = re.search(r'COVERAGE:\s*([0-9.]+)', content, re.IGNORECASE)
        sufficient_match = re.search(r'SUFFICIENT:\s*(YES|NO)', content, re.IGNORECASE)
        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE)
        
        # Blend LLM score with baseline
        llm_relevance = float(relevance_match.group(1)) if relevance_match else 0.5
        final_relevance = max(baseline_relevance, llm_relevance)  # Take the better score
        
        coverage = float(coverage_match.group(1)) if coverage_match else 0.3
        sufficient = sufficient_match.group(1).upper() == 'YES' if sufficient_match else False
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
        
        # AUTO-PASS if relevance is decent
        if final_relevance >= 0.5 and coverage >= 0.3:
            sufficient = True
        
        result = RetrievalEvaluation(
            is_sufficient=sufficient,
            relevance_score=final_relevance,
            coverage_score=coverage,
            should_retry=not sufficient,
            retry_suggestion="",
            reasoning=reasoning
        )
        
        print(f"    Relevance: {result.relevance_score:.2f} (baseline: {baseline_relevance:.2f})")
        print(f"    Coverage: {result.coverage_score:.2f}")
        print(f"    Sufficient: {result.is_sufficient}")
        
        return result

# Quick test
if __name__ == "__main__":
    from langchain.schema import Document
    
    agent = RetrievalEvaluatorAgent()
    
    # Mock documents
    mock_docs = [
        Document(
            page_content="GPT-3 is a large language model with 175 billion parameters...",
            metadata={'title': 'GPT-3 Paper'}
        ),
        Document(
            page_content="The transformer architecture uses attention mechanisms...",
            metadata={'title': 'Attention Paper'}
        )
    ]
    
    query = "How many parameters does GPT-3 have?"
    evaluation = agent.evaluate(query, mock_docs)
    
    print(f"\nReasoning: {evaluation.reasoning}")