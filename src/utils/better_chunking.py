"""
Smart chunking strategies for different types of papers
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


class SmartChunker:
    """
    Intelligent chunking that adapts to document importance and structure
    """
    
    def __init__(self):
        # Small chunks for famous papers (need precision)
        self.small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Medium chunks for general content
        self.medium_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Large chunks for context-heavy sections
        self.large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def detect_section_type(self, text: str) -> str:
        """
        Detect what type of section this is
        Returns: 'abstract', 'introduction', 'methods', 'results', 'references', 'body'
        """
        text_lower = text.lower()
        first_100 = text_lower[:100]
        
        # Check for abstract
        if 'abstract' in first_100:
            return 'abstract'
        
        # Check for introduction
        if any(kw in first_100 for kw in ['introduction', '1. introduction', '1 introduction']):
            return 'introduction'
        
        # Check for methods
        if any(kw in first_100 for kw in ['method', 'approach', 'model architecture']):
            return 'methods'
        
        # Check for results
        if any(kw in first_100 for kw in ['result', 'experiment', 'evaluation']):
            return 'results'
        
        # Check for references
        if 'references' in first_100 or text.count('[1]') > 5:
            return 'references'
        
        return 'body'
    
    def is_important_content(self, text: str) -> bool:
        """
        Check if this text contains important information
        """
        # Check for key phrases
        important_phrases = [
            'we present', 'we propose', 'we introduce',
            'our main contribution', 'our method',
            'parameters', 'architecture', 'model',
            'billion', 'million', 'layers', 'attention heads',
            'training', 'dataset'
        ]
        
        text_lower = text.lower()
        phrase_count = sum(1 for phrase in important_phrases if phrase in text_lower)
        
        # If mentions multiple key phrases, it's important
        return phrase_count >= 2
    
    def chunk_document(self, doc: Document) -> List[Document]:
        """
        Intelligently chunk a document based on its category and content
        """
        category = doc.metadata.get('category', 'recent')
        text = doc.page_content
        
        print(f"    Chunking: {doc.metadata.get('title', 'Unknown')[:40]}...")
        print(f"    Category: {category}")
        
        # Famous papers get special treatment
        if category == 'famous':
            chunks = self._chunk_famous_paper(doc)
        else:
            chunks = self._chunk_regular_paper(doc)
        
        print(f"      Created {len(chunks)} chunks")
        
        return chunks
    
    def _chunk_famous_paper(self, doc: Document) -> List[Document]:
        """
        Chunk famous papers with extra care
        """
        text = doc.page_content
        all_chunks = []
        
        # Split into rough sections (by double newlines)
        sections = text.split('\n\n')
        
        current_section_text = ""
        current_section_type = "body"
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip tiny sections
                continue
            
            # Detect section type
            section_type = self.detect_section_type(section)
            
            # If section type changed, process accumulated text
            if section_type != current_section_type and current_section_text:
                chunks = self._process_section(
                    current_section_text, 
                    current_section_type,
                    doc.metadata,
                    is_famous=True
                )
                all_chunks.extend(chunks)
                current_section_text = ""
            
            current_section_text += section + "\n\n"
            current_section_type = section_type
        
        # Process remaining text
        if current_section_text:
            chunks = self._process_section(
                current_section_text,
                current_section_type,
                doc.metadata,
                is_famous=True
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_regular_paper(self, doc: Document) -> List[Document]:
        """
        Chunk regular papers with medium chunks
        """
        chunks = self.medium_splitter.split_documents([doc])
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['is_important'] = self.is_important_content(chunk.page_content)
        
        return chunks
    
    def _process_section(self, text: str, section_type: str, 
                        base_metadata: dict, is_famous: bool) -> List[Document]:
        """
        Process a section with appropriate chunking strategy
        """
        # Choose splitter based on section importance
        if section_type in ['abstract', 'introduction']:
            splitter = self.small_splitter  # Small chunks for important sections
        elif section_type == 'references':
            # Don't chunk references much - less useful
            return []  # Skip references entirely
        else:
            splitter = self.medium_splitter if is_famous else self.large_splitter
        
        # Create document
        temp_doc = Document(page_content=text, metadata=base_metadata.copy())
        
        # Split
        chunks = splitter.split_documents([temp_doc])
        
        # Add section metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['section_type'] = section_type
            chunk.metadata['is_important'] = (
                section_type in ['abstract', 'introduction'] or 
                self.is_important_content(chunk.page_content)
            )
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks


# Quick test
if __name__ == "__main__":
    from langchain.schema import Document
    
    chunker = SmartChunker()
    
    # Test document
    test_doc = Document(
        page_content="""
        Abstract

        We present GPT-3, a language model with 175 billion parameters. 
        The model was trained on 300 billion tokens.

        1. Introduction

        Large language models have shown impressive capabilities...
        """,
        metadata={
            'title': 'GPT-3 Paper',
            'category': 'famous'
        }
    )
    
    chunks = chunker.chunk_document(test_doc)
    
    print(f"\nResults:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Section: {chunk.metadata.get('section_type')}")
        print(f"  Important: {chunk.metadata.get('is_important')}")
        print(f"  Text: {chunk.page_content[:100]}...")