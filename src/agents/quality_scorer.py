"""
Agent 5
Quality Scorer
Provides overall quality assessment of the answer
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class QualityScore(BaseModel):
    """Overall quality assessment"""
    overall_score: float = Field(description="Overall quality 0-1")
    factual_accuracy: float = Field(description="Factual correctness 0-1")
    citation_quality: float = Field(description="Citation quality 0-1")
    relevance: float = Field(description="Relevance to query 0-1")
    completeness: float = Field(description="Answer completeness 0-1")
    clarity: float = Field(description="Writing clarity 0-1")
    recommendation: str = Field(description="PASS, FLAG, or REGENERATE")
    feedback: str = Field(description="Specific feedback")

class QualityScorerAgent:
    """Agent that provides overall quality assessment"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
    
    def score(self, query: str, answer: str, fact_check_result, retrieval_eval) -> QualityScore:
        """Score the overall quality"""
        
        print(f"\n[Quality Scorer] Evaluating overall quality...")
        
        # Build assessment prompt
        prompt = f"""Evaluate the overall quality of this RAG system output.

        Original Query: {query}

        Generated Answer:
        {answer}

        Fact-Check Results:
        - Total claims: {fact_check_result.total_claims}
        - Verified: {fact_check_result.verified_claims}
        - Failed: {fact_check_result.failed_claims}
        - Avg confidence: {fact_check_result.average_confidence:.2f}

        Retrieval Quality:
        - Relevance: {retrieval_eval.relevance_score:.2f}
        - Coverage: {retrieval_eval.coverage_score:.2f}

        Rate the answer on:
        1. Factual Accuracy (0-1): Based on fact-check results
        2. Citation Quality (0-1): Are sources properly cited?
        3. Relevance (0-1): Does it answer the query?
        4. Completeness (0-1): Is it thorough?
        5. Clarity (0-1): Is it well-written?

        Then provide:
        - Overall Score (0-1): Weighted average
        - Recommendation: PASS (>0.8), FLAG (0.6-0.8), or REGENERATE (<0.6)
        - Feedback: Specific improvement suggestions

        Format:
        Factual Accuracy: [0.0-1.0]
        Citation Quality: [0.0-1.0]
        Relevance: [0.0-1.0]
        Completeness: [0.0-1.0]
        Clarity: [0.0-1.0]
        Overall Score: [0.0-1.0]
        Recommendation: [PASS/FLAG/REGENERATE]
        Feedback: [specific feedback]
        """
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Parse scores
        import re
        
        def extract_score(pattern, text):
            match = re.search(pattern, text, re.IGNORECASE)
            return float(match.group(1)) if match else 0.5
        
        factual = extract_score(r'factual accuracy:\s*([0-9.]+)', content)
        citation = extract_score(r'citation quality:\s*([0-9.]+)', content)
        relevance = extract_score(r'relevance:\s*([0-9.]+)', content)
        completeness = extract_score(r'completeness:\s*([0-9.]+)', content)
        clarity = extract_score(r'clarity:\s*([0-9.]+)', content)
        overall = extract_score(r'overall score:\s*([0-9.]+)', content)
        
        # Extract recommendation
        if 'pass' in content.lower():
            recommendation = 'PASS'
        elif 'regenerate' in content.lower():
            recommendation = 'REGENERATE'
        else:
            recommendation = 'FLAG'
        
        # Extract feedback
        feedback_match = re.search(r'feedback:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else "No specific feedback"
        
        result = QualityScore(
            overall_score=overall,
            factual_accuracy=factual,
            citation_quality=citation,
            relevance=relevance,
            completeness=completeness,
            clarity=clarity,
            recommendation=recommendation,
            feedback=feedback
        )
        
        print(f"Overall Score: {result.overall_score:.2f}")
        print(f"Recommendation: {result.recommendation}")
        
        return result

if __name__ == "__main__":
    from fact_checker import FactCheckResult, Claim
    from retrieval_evaluator import RetrievalEvaluation
    
    agent = QualityScorerAgent()
    
    # Mock data
    mock_fact_check = FactCheckResult(
        claims=[],
        total_claims=3,
        verified_claims=3,
        failed_claims=0,
        average_confidence=0.92,
        has_issues=False
    )
    
    mock_retrieval = RetrievalEvaluation(
        is_sufficient=True,
        relevance_score=0.88,
        coverage_score=0.85,
        should_retry=False,
        retry_suggestion="",
        reasoning="Good retrieval"
    )
    
    query = "How many parameters does GPT-3 have?"
    answer = "GPT-3 has 175 billion parameters [1]."
    
    score = agent.score(query, answer, mock_fact_check, mock_retrieval)
    print(f"\nFeedback: {score.feedback}")