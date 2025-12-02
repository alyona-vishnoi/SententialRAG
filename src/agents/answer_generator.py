"""
Agent 3
Answer Generator
Generates answers with inline citations
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class GeneratedAnswer(BaseModel):
    """Generated answer with metadata"""
    answer: str = Field(description="The generated answer with citations")
    num_citations: int = Field(description="Number of citations used")
    sources_used: list[str] = Field(description="List of source titles used")

class AnswerGeneratorAgent:
    """Agent that generates well-cited answers"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3  # Slightly creative but still grounded
        )
    
    def generate(self, query: str, retrieved_docs: list) -> GeneratedAnswer:
        """Generate answer with citations"""
        
        print(f"\n[Answer Generator] Creating answer...")
        
        # Build context with numbered sources
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.metadata.get('title', f'Document {i}')
            sources.append(title)
            context_parts.append(f"[{i}] {title}:\n{doc.page_content}\n")
        
        context = '\n'.join(context_parts)
        
        prompt = f"""Answer the question using ONLY the provided context.

        CRITICAL RULES:
        1. Cite sources using [1], [2], etc. after EVERY factual claim
        2. If information is in source 1, write: "claim [1]"
        3. Use multiple citations if claim appears in multiple sources: "claim [1][2]"
        4. DO NOT make claims without citations
        5. If you cannot answer from context, say "The provided sources do not contain..."

        Context:
        {context}

        Question: {query}

        Provide a clear answer with inline citations:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Count citations
        import re
        citations = re.findall(r'\[\d+\]', answer)
        num_citations = len(set(citations))  # Unique citations
        
        result = GeneratedAnswer(
            answer=answer,
            num_citations=num_citations,
            sources_used=sources
        )
        
        print(f"Answer generated ({len(answer)} chars)")
        print(f"Citations: {num_citations}")
        
        return result

# Quick test
if __name__ == "__main__":
    from langchain.schema import Document
    
    agent = AnswerGeneratorAgent()
    
    mock_docs = [
        Document(
            page_content="GPT-3, introduced by Brown et al., has 175 billion parameters and was trained on 300 billion tokens.",
            metadata={'title': 'Language Models are Few-Shot Learners'}
        ),
        Document(
            page_content="The model uses a transformer architecture with 96 attention layers.",
            metadata={'title': 'GPT-3 Technical Report'}
        )
    ]
    
    query = "How many parameters does GPT-3 have?"
    result = agent.generate(query, mock_docs)
    
    print(f"\nGenerated Answer:\n{result.answer}")
    print(f"\nSources: {result.sources_used}")