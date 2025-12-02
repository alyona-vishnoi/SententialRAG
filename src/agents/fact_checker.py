"""
Agent 4
Fact Checker
Verifies every claim in the answer against source documents
"""

import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class Claim(BaseModel):
    """A single factual claim"""
    text: str
    citation: str  # e.g., "[1]"
    verified: bool
    confidence: float
    evidence: str = ""

class FactCheckResult(BaseModel):
    """Result of fact-checking"""
    claims: list[Claim]
    total_claims: int
    verified_claims: int
    failed_claims: int
    average_confidence: float
    has_issues: bool
    issues_description: str = ""

class FactCheckerAgent:
    """Agent that verifies factual claims"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0  # Very deterministic for fact-checking
        )
    
    def extract_claims(self, answer: str) -> list[dict]:
        """Extract individual claims from the answer"""
        
        # Split by sentences (improved regex)
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract citations
            citations = re.findall(r'\[\d+\]', sentence)
            
            # Remove citations from text for analysis
            clean_text = re.sub(r'\[\d+\]', '', sentence).strip()
            
            if clean_text and len(clean_text) > 10:  # Ignore very short fragments
                claims.append({
                    'text': clean_text,
                    'citations': citations,
                    'original': sentence
                })
        
        return claims
    
    def verify_claim(self, claim_text: str, citation: str, source_docs: list) -> Claim:
        """Verify a single claim against sources"""
        
        # Get the source document
        try:
            source_num = int(citation.strip('[]'))
            if source_num <= len(source_docs):
                source_doc = source_docs[source_num - 1]
                source_text = source_doc.page_content
                source_title = source_doc.metadata.get('title', f'Source {source_num}')
            else:
                # Invalid citation number
                return Claim(
                    text=claim_text,
                    citation=citation,
                    verified=False,
                    confidence=0.0,
                    evidence="Citation number exceeds available sources"
                )
        except:
            return Claim(
                text=claim_text,
                citation=citation,
                verified=False,
                confidence=0.0,
                evidence="Invalid citation format"
            )
        
        # Use LLM to verify the claim - IMPROVED PROMPT
        prompt = f"""You are a fact-checker. Verify if the MEANING of the claim is supported.
            CLAIM TO VERIFY:
            "{claim_text}"

            SOURCE DOCUMENT ({source_title}):
            {source_text[:2000]}

            RULES:
            1. PARAPHRASES ARE OK - If claim and source say the same thing in different words, mark as VERIFIED
            2. NUMBERS MUST BE EXACT - "175 billion" vs "200 billion" = NOT VERIFIED
            3. PARTIAL MATCHES COUNT - If claim is partially supported, give confidence 0.6-0.8

            Examples:
            Claim: "Transformers are an alternative to RNNs"
            Source: "aligned RNNs or convolution"
            -> VERIFIED: YES, CONFIDENCE: 0.85 (paraphrase is OK)

            Claim: "GPT-3 has 200B parameters"
            Source: "175 billion parameters"
            -> VERIFIED: NO, CONFIDENCE: 0.2 (wrong number)

            RESPOND:
            VERIFIED: [YES or NO]
            CONFIDENCE: [0.0-1.0]
            EVIDENCE: [quote from source]
        """
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        print(f"\nFact-check response:\n{content[:200]}...")
        
        # Parse response - MORE ROBUST
        verified = False
        confidence = 0.0
        evidence = ""
        
        # Check verified status
        verified_match = re.search(r'VERIFIED:\s*(YES|NO)', content, re.IGNORECASE)
        if verified_match:
            verified = verified_match.group(1).upper() == 'YES'
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5
        
        # If verified=NO but confidence is high, fix it
        if not verified and confidence > 0.5:
            confidence = 0.3  # Force low confidence for failed verification
        
        # If verified=YES but confidence is low, that's suspicious
        if verified and confidence < 0.5:
            verified = False  # Don't trust it
        
        # Extract evidence
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if evidence_match:
            evidence = evidence_match.group(1).strip()
        else:
            evidence = content[:150]
        
        return Claim(
            text=claim_text,
            citation=citation,
            verified=verified,
            confidence=confidence,
            evidence=evidence
        )
    
    def check(self, answer: str, source_docs: list) -> FactCheckResult:
        """Fact-check the entire answer"""
        
        print(f"\n[Fact Checker] Verifying claims...")
        
        # Extract claims
        extracted = self.extract_claims(answer)
        print(f"Found {len(extracted)} claims to verify")
        
        verified_claims = []
        
        for i, claim_info in enumerate(extracted, 1):
            claim_text = claim_info['text']
            citations = claim_info['citations']
            
            print(f"\n   Claim {i}: {claim_text[:60]}...")
            
            if not citations:
                # No citation - automatic fail
                verified_claims.append(Claim(
                    text=claim_text,
                    citation="",
                    verified=False,
                    confidence=0.0,
                    evidence="No citation provided"
                ))
                print(f"No citation provided")
                continue
            
            # Check against first citation
            citation = citations[0]
            
            result = self.verify_claim(claim_text, citation, source_docs)
            verified_claims.append(result)
            
            status = "✅" if result.verified else "❌"
            print(f" {status} Verified: {result.verified}, Confidence: {result.confidence:.2f}")
            print(f" Evidence: {result.evidence[:100]}...")
        
        # Calculate statistics
        total = len(verified_claims)
        verified_count = sum(1 for c in verified_claims if c.verified)
        failed_count = total - verified_count
        avg_conf = sum(c.confidence for c in verified_claims) / total if total > 0 else 0.0
        
        has_issues = failed_count > 0 or avg_conf < 0.7
        
        issues_desc = ""
        if failed_count > 0:
            issues_desc = f"{failed_count}/{total} claims could not be verified"
        
        result = FactCheckResult(
            claims=verified_claims,
            total_claims=total,
            verified_claims=verified_count,
            failed_claims=failed_count,
            average_confidence=avg_conf,
            has_issues=has_issues,
            issues_description=issues_desc
        )
        
        print(f"\nFINAL: {verified_count}/{total} verified, {avg_conf:.2%} avg confidence")
        if has_issues:
            print(f"ISSUES: {issues_desc}")
        
        return result

# quick test
if __name__ == "__main__":
    from langchain.schema import Document
    
    print("="*60)
    print("FACT CHECKER TEST SUITE")
    print("="*60)
    
    agent = FactCheckerAgent()
    
    # Create a detailed test document
    test_doc = Document(
        page_content="""
        GPT-3, introduced by Brown et al. in 2020, has 175 billion parameters. 
        The model was trained on 300 billion tokens from diverse sources including 
        Common Crawl, WebText2, Books1, Books2, and Wikipedia. The largest model 
        in the GPT-3 family has 175 billion parameters and uses 96 attention layers.
        Training used approximately 3.14E23 FLOPs.
        """,
        metadata={'title': 'Language Models are Few-Shot Learners (GPT-3 Paper)'}
    )
    
    print("\nTest Document Content (summary):")
    print("   - GPT-3 has 175 billion parameters")
    print("   - Trained on 300 billion tokens")
    print("   - Uses 96 attention layers")
    print("   - Published by Brown et al. in 2020")
    
    # TEST 1: Correct fact
    print("\n" + "="*60)
    print("TEST 1: CORRECT FACT")
    print("="*60)
    answer1 = "GPT-3 has 175 billion parameters [1]."
    print(f"Answer: {answer1}")
    result1 = agent.check(answer1, [test_doc])
    print(f"\nExpected: VERIFIED=True, High Confidence")
    print(f"Got: VERIFIED={result1.claims[0].verified}, CONFIDENCE={result1.claims[0].confidence:.2f}")
    
    # TEST 2: Incorrect fact (wrong number)
    print("\n" + "="*60)
    print("TEST 2: INCORRECT FACT (Wrong Number)")
    print("="*60)
    answer2 = "GPT-3 has 200 billion parameters [1]."
    print(f"Answer: {answer2}")
    result2 = agent.check(answer2, [test_doc])
    print(f"\nExpected: VERIFIED=False, Low Confidence")
    print(f"Got: VERIFIED={result2.claims[0].verified}, CONFIDENCE={result2.claims[0].confidence:.2f}")
    
    # TEST 3: Correct fact about training
    print("\n" + "="*60)
    print("TEST 3: CORRECT FACT (Training Data)")
    print("="*60)
    answer3 = "GPT-3 was trained on 300 billion tokens [1]."
    print(f"Answer: {answer3}")
    result3 = agent.check(answer3, [test_doc])
    print(f"\nExpected: VERIFIED=True, High Confidence")
    print(f"Got: VERIFIED={result3.claims[0].verified}, CONFIDENCE={result3.claims[0].confidence:.2f}")
    
    # TEST 4: Completely made up fact
    print("\n" + "="*60)
    print("TEST 4: MADE UP FACT")
    print("="*60)
    answer4 = "GPT-3 was trained in 2025 and uses quantum computing [1]."
    print(f"Answer: {answer4}")
    result4 = agent.check(answer4, [test_doc])
    print(f"\nExpected: VERIFIED=False, Very Low Confidence")
    print(f"Got: VERIFIED={result4.claims[0].verified}, CONFIDENCE={result4.claims[0].confidence:.2f}")
    
    # TEST 5: Multiple claims (mixed)
    print("\n" + "="*60)
    print("TEST 5: MULTIPLE CLAIMS (One Correct, One Wrong)")
    print("="*60)
    answer5 = "GPT-3 has 175 billion parameters [1]. It was trained on 500 billion tokens [1]."
    print(f"Answer: {answer5}")
    result5 = agent.check(answer5, [test_doc])
    print(f"\nExpected: 1 verified, 1 failed")
    print(f"Got: {result5.verified_claims}/{result5.total_claims} verified")
    for i, claim in enumerate(result5.claims, 1):
        status = "✅" if claim.verified else "❌"
        print(f" {status} Claim {i}: {claim.text[:50]}... (conf: {claim.confidence:.2f})")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("The fact checker should:")
    print("Give high confidence (>0.8) for correct facts")
    print("Give low confidence (<0.4) for incorrect facts")
    print("Show clear verified=True/False status")