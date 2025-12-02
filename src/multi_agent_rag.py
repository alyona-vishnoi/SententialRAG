"""
Multi-Agent RAG System using LangGraph
Orchestrates 5 specialized agents
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from utils.hybrid_retriever import HybridRetriever

# Import our agents
from agents.query_analyzer import QueryAnalyzerAgent
from agents.retrieval_evaluator import RetrievalEvaluatorAgent
from agents.answer_generator import AnswerGeneratorAgent
from agents.fact_checker import FactCheckerAgent
from agents.quality_scorer import QualityScorerAgent

# Import state
from graph_state import RAGState

load_dotenv()


class MultiAgentRAG:
    """
    Production-grade RAG system with automated quality assurance
    Uses 5 specialized agents orchestrated with LangGraph
    """
    
    def __init__(self, vector_store_path: str = "data/chroma_db"):
        print("Initializing Multi-Agent RAG System...")
        print("="*60)
        
        # Initialize agents
        print("Loading agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retrieval_evaluator = RetrievalEvaluatorAgent()
        self.answer_generator = AnswerGeneratorAgent()
        self.fact_checker = FactCheckerAgent()
        self.quality_scorer = QualityScorerAgent()
        print("All agents loaded!")
        
        # Initialize vector store
        print("Loading vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        print("Vector store ready!")

        # Initialize hybrid retriever
        print("Initializing hybrid retriever...")
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        print("Hybrid retriever ready!")
        
        # Build the graph
        print("Building agent graph...")
        self.graph = self._build_graph()
        print("Graph built!")
        
        print("\n Multi-Agent RAG System Ready!")
        print("="*60)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes (each node is an agent)
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("evaluate_retrieval", self.evaluate_retrieval_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("fact_check", self.fact_check_node)
        workflow.add_node("score_quality", self.score_quality_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        # analyze_query -> retrieve_documents
        workflow.add_edge("analyze_query", "retrieve_documents")
        
        # retrieve_documents -> evaluate_retrieval
        workflow.add_edge("retrieve_documents", "evaluate_retrieval")
        
        # evaluate_retrieval -> (conditional)
        workflow.add_conditional_edges(
            "evaluate_retrieval",
            self.should_retry_retrieval,
            {
                "retry": "retrieve_documents",  # Try again with better query
                "continue": "generate_answer"    # Good enough, continue
            }
        )
        
        # generate_answer -> fact_check
        workflow.add_edge("generate_answer", "fact_check")
        
        # fact_check -> score_quality
        workflow.add_edge("fact_check", "score_quality")
        
        # score_quality -> END
        workflow.add_edge("score_quality", END)
        
        return workflow.compile()
    
    # ========== NODE FUNCTIONS ==========
    
    def analyze_query_node(self, state: RAGState) -> RAGState:
        """Node: Analyze the user's query"""
        print("\n" + "="*60)
        print("STEP 1: Query Analysis")
        print("="*60)
        
        analysis = self.query_analyzer.analyze(state["query"])
        
        state["query_type"] = analysis.query_type
        state["key_entities"] = analysis.key_entities
        state["optimized_query"] = analysis.search_query
        state["requires_multi_doc"] = analysis.requires_multi_doc
        state["retrieval_attempts"] = 0
        
        return state
    
    def retrieve_documents_node(self, state: RAGState) -> RAGState:
        """Node: Retrieve relevant documents"""
        print("\n" + "="*60)
        print("STEP 2: Document Retrieval")
        print("="*60)
        
        # Determine number of docs based on query type
        num_docs = 5 if state["requires_multi_doc"] else 3
        
        # Use optimized query
        query = state["optimized_query"]
        print(f"Searching for: '{query}'")
        print(f"Retrieving top {num_docs} documents...")
        
        # Retrieve
        # docs = self.vector_store.similarity_search(query, k=num_docs)
        docs = self.hybrid_retriever.retrieve(query, k=num_docs)
        
        state["retrieved_docs"] = docs
        state["retrieval_attempts"] += 1
        
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Unknown')[:50]
            is_important = doc.metadata.get('is_important', False)
            marker = "*important*" if is_important else "  "
            print(f" {marker}[{i}] {title}...")
            print(f" [{i}] {title}...")
        
        return state
    
    def evaluate_retrieval_node(self, state: RAGState) -> RAGState:
        """Node: Evaluate retrieval quality"""
        print("\n" + "="*60)
        print("STEP 3: Retrieval Evaluation")
        print("="*60)
        
        evaluation = self.retrieval_evaluator.evaluate(
            state["query"],
            state["retrieved_docs"]
        )
        
        state["retrieval_relevance"] = evaluation.relevance_score
        state["retrieval_coverage"] = evaluation.coverage_score
        state["should_retry_retrieval"] = evaluation.should_retry
        
        # Only retry once
        if state["retrieval_attempts"] >= 2:
            state["should_retry_retrieval"] = False
            print("Max retrieval attempts reached, proceeding...")
        
        return state
    
    def should_retry_retrieval(self, state: RAGState) -> str:
        """Conditional edge: Should we retry retrieval?"""
        
        # Check if we've already retried
        if state["retrieval_attempts"] >= 2:
            print("! Max retries reached - proceeding with current results")
            return "continue"
        
        # Check if retrieval was good enough
        if state["should_retry_retrieval"]:
            # Only retry if relevance is VERY low (< 0.3)
            if state["retrieval_relevance"] < 0.3:
                print("Retrieval quality very low - retrying...")
                return "retry"
            else:
                print("Retrieval marginal but acceptable - proceeding...")
                return "continue"
        else:
            print("Retrieval quality good - continuing...")
            return "continue"
    
    def generate_answer_node(self, state: RAGState) -> RAGState:
        """Node: Generate answer with citations"""
        print("\n" + "="*60)
        print("STEP 4: Answer Generation")
        print("="*60)
        
        result = self.answer_generator.generate(
            state["query"],
            state["retrieved_docs"]
        )
        
        state["answer"] = result.answer
        state["num_citations"] = result.num_citations
        state["sources_used"] = result.sources_used
        
        return state
    
    def fact_check_node(self, state: RAGState) -> RAGState:
        """Node: Fact-check the answer"""
        print("\n" + "="*60)
        print("STEP 5: Fact Checking")
        print("="*60)
        
        result = self.fact_checker.check(
            state["answer"],
            state["retrieved_docs"]
        )
        
        state["total_claims"] = result.total_claims
        state["verified_claims"] = result.verified_claims
        state["failed_claims"] = result.failed_claims
        state["fact_check_confidence"] = result.average_confidence
        state["has_fact_issues"] = result.has_issues
        state["fact_issues_desc"] = result.issues_description
        
        return state
    
    def score_quality_node(self, state: RAGState) -> RAGState:
        """Node: Score overall quality"""
        print("\n" + "="*60)
        print("STEP 6: Quality Scoring")
        print("="*60)
        
        # Need to recreate the fact check result object for the scorer
        from agents.fact_checker import FactCheckResult
        from agents.retrieval_evaluator import RetrievalEvaluation
        
        fact_check_result = FactCheckResult(
            claims=[],  # We don't need individual claims here
            total_claims=state["total_claims"],
            verified_claims=state["verified_claims"],
            failed_claims=state["failed_claims"],
            average_confidence=state["fact_check_confidence"],
            has_issues=state["has_fact_issues"],
            issues_description=state["fact_issues_desc"]
        )
        
        retrieval_eval = RetrievalEvaluation(
            is_sufficient=True,
            relevance_score=state["retrieval_relevance"],
            coverage_score=state["retrieval_coverage"],
            should_retry=False,
            retry_suggestion="",
            reasoning=""
        )
        
        score = self.quality_scorer.score(
            state["query"],
            state["answer"],
            fact_check_result,
            retrieval_eval
        )
        
        state["overall_score"] = score.overall_score
        state["factual_accuracy"] = score.factual_accuracy
        state["citation_quality"] = score.citation_quality
        state["relevance"] = score.relevance
        state["completeness"] = score.completeness
        state["recommendation"] = score.recommendation
        state["feedback"] = score.feedback
        state["final_status"] = "SUCCESS" if score.recommendation == "PASS" else "NEEDS_REVIEW"
        
        return state
    
    # ========== PUBLIC API ==========
    
    def query(self, question: str) -> dict:
        """
        Main entry point: Process a query through the multi-agent system
        
        Args:
            question: User's question
            
        Returns:
            dict with answer and quality metrics
        """
        
        print("\n" + "~"*30)
        print(f"QUERY: {question}")
        print("~"*30)
        
        # Initialize state
        initial_state = {
            "query": question,
            "retrieval_attempts": 0,
            "should_retry_retrieval": False,
            "should_regenerate": False,
        }
        
        # Run the graph!
        final_state = self.graph.invoke(initial_state)
        
        # Print final results
        self._print_results(final_state)
        
        return final_state
    
    def _print_results(self, state: RAGState):
        """Pretty print the final results"""
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        print(f"\nQuestion: {state['query']}")
        
        print(f"\nAnswer:")
        print(f"{state['answer']}")
        
        print(f"\nSources Used:")
        for i, source in enumerate(state['sources_used'], 1):
            print(f"   [{i}] {source}")
        
        print(f"\nQuality Metrics:")
        print(f"   Overall Score:      {state['overall_score']:.2f}")
        print(f"   Factual Accuracy:   {state['factual_accuracy']:.2f}")
        print(f"   Citation Quality:   {state['citation_quality']:.2f}")
        print(f"   Relevance:          {state['relevance']:.2f}")
        print(f"   Completeness:       {state['completeness']:.2f}")
        
        print(f"\nFact-Check Results:")
        print(f"   Total Claims:       {state['total_claims']}")
        print(f"   Verified:           {state['verified_claims']}")
        print(f"   Failed:             {state['failed_claims']}")
        print(f"   Confidence:         {state['fact_check_confidence']:.2%}")
        
        print(f"\nRecommendation: {state['recommendation']}")
        
        if state['feedback']:
            print(f"\nFeedback: {state['feedback']}")
        
        print("\n" + "="*60)


# ========== DEMO ==========

def main():
    """Demo the multi-agent RAG system"""
    
    # Initialize
    rag = MultiAgentRAG()
    
    # Test questions
    test_questions = [
        "How many parameters does GPT-3 have?",
        "What is the architecture of BERT?",
        "Compare transformers and RNNs",
    ]
    
    print("\n\n" + "🎬 "*30)
    print("STARTING DEMO")
    print("🎬 "*30)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'='*60}")
        print(f"QUESTION {i}/{len(test_questions)}")
        print(f"{'='*60}")
        
        result = rag.query(question)
        results.append(result)
        
        if i < len(test_questions):
            input("\n'->Press Enter for next question...")
    
    # Summary
    print("\n\n" + "📊 "*30)
    print("DEMO SUMMARY")
    print("📊 "*30)
    
    for i, (q, r) in enumerate(zip(test_questions, results), 1):
        print(f"\n{i}. {q}")
        print(f"   Score: {r['overall_score']:.2f} | {r['recommendation']}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()