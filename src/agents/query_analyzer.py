"""
Agent 1
Query Analyzer
Understands user intent and extracts key information for better retrieval
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


class QueryAnalysis(BaseModel):
    """Structured output from query analysis"""
    query_type: str = Field(description="Type: factual, comparative, temporal, conceptual")
    key_entities: list[str] = Field(description="Key entities mentioned (models, papers, authors)")
    search_query: str = Field(description="Optimized query for vector search")
    requires_multi_doc: bool = Field(description="Does this need multiple documents?")
    requires_calculation: bool = Field(description="Does this need synthesis/calculation?")

class QueryAnalyzerAgent:
    """Agent that analyzes and understands user queries"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1  # Low temperature for analytical tasks
        )
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze the user's query"""
        
        print(f"\n[Query Analyzer] Analyzing: '{query}'")
        
        prompt = f"""Analyze this user query and extract key information.

        Query: {query}

        Provide:
        1. Query type (factual, comparative, temporal, conceptual)
        2. Key entities mentioned (model names, paper titles, authors, etc.)
        3. An optimized search query for finding relevant documents
        4. Whether multiple documents are needed
        5. Whether this requires calculation or synthesis

        Format your response as:
        Query Type: [type]
        Key Entities: [entity1, entity2, ...]
        Search Query: [optimized query]
        Multi-Document: [yes/no]
        Requires Calculation: [yes/no]

        Example:
        Query: "Compare GPT-3 and BERT architectures"
        Query Type: comparative
        Key Entities: GPT-3, BERT, architecture
        Search Query: GPT-3 BERT architecture transformer
        Multi-Document: yes
        Requires Calculation: no
        """
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Parse the response
        lines = content.strip().split('\n')
        analysis = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('-', '_')
                value = value.strip()
                analysis[key] = value
        
        # Extract entities as list
        entities_str = analysis.get('key_entities', '')
        entities = [e.strip() for e in entities_str.split(',') if e.strip()]
        
        result = QueryAnalysis(
            query_type=analysis.get('query_type', 'factual'),
            key_entities=entities,
            search_query=analysis.get('search_query', query),
            requires_multi_doc=analysis.get('multi_document', 'no').lower() == 'yes',
            requires_calculation=analysis.get('requires_calculation', 'no').lower() == 'yes'
        )
        
        print(f"Type: {result.query_type}")
        print(f"Entities: {result.key_entities}")
        print(f"Search query: {result.search_query}")
        print(f"Multi-doc: {result.requires_multi_doc}")
        
        return result

# Quick test
if __name__ == "__main__":
    agent = QueryAnalyzerAgent()
    
    test_queries = [
        "How many parameters does GPT-3 have?",
        "Compare BERT and GPT architectures",
        "What are recent advances in transformers?"
    ]
    
    for query in test_queries:
        analysis = agent.analyze(query)
        print(f"\n{'-'*60}\n")