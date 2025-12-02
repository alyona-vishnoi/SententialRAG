"""
Download famous papers AND recent papers for a comprehensive dataset
"""

import arxiv
import json
from pathlib import Path
from PyPDF2 import PdfReader

def download_famous_papers():
    """Download specific famous ML papers"""
    
    famous_papers = {
        "1706.03762": "Attention Is All You Need (Transformers) - 2017",
        "1810.04805": "BERT - 2018",
        "2005.14165": "GPT-3 - 2020",
        "2010.11929": "Vision Transformer (ViT) - 2020",
        "1512.03385": "ResNet - 2015",
        "1409.0473": "GoogLeNet (Inception) - 2014",
        "1409.1556": "VGG - 2014",
    }
    
    print("DOWNLOADING FAMOUS PAPERS")
    print("="*60)
    
    papers_info = []
    
    for paper_id, description in famous_papers.items():
        try:
            print(f"\n{description}")
            
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            print(f"   Title: {paper.title}")
            
            papers_info.append({
                'title': paper.title,
                'arxiv_id': paper_id,
                'description': description,
                'authors': [a.name for a in paper.authors],
                'published': paper.published.strftime('%Y-%m-%d'),
                'category': 'famous',
                'paper': paper 
            })
            
            print(f"Added to download queue")
            
        except Exception as e:
            print(f"Error: {e}")
    
    return papers_info

def download_recent_papers(num_papers=20):
    """Download recent ML papers"""
    
    print("\nDOWNLOADING RECENT PAPERS")
    print("="*60)
    
    # Search for recent ML papers
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI OR cat:cs.CL",  # ML, AI, NLP categories
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers_info = []
    
    for i, paper in enumerate(search.results(), 1):
        print(f"\n[{i}/{num_papers}] {paper.title[:60]}...")
        
        papers_info.append({
            'title': paper.title,
            'arxiv_id': paper.entry_id.split('/')[-1],
            'authors': [a.name for a in paper.authors],
            'published': paper.published.strftime('%Y-%m-%d'),
            'summary': paper.summary,
            'category': 'recent',
            'paper': paper
        })
        
        print(f"Added to download queue")
    
    return papers_info

def download_and_process_all(famous_papers_list, recent_papers_list):
    """Download PDFs and process all papers"""
    
    data_dir = Path("data/papers")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    all_papers = famous_papers_list + recent_papers_list
    
    print(f"\nDOWNLOADING AND PROCESSING {len(all_papers)} PAPERS")
    print("="*60)
    
    processed_documents = []
    metadata_list = []
    
    for i, paper_info in enumerate(all_papers, 1):
        try:
            paper = paper_info['paper']
            arxiv_id = paper_info['arxiv_id']
            category = paper_info['category']
            
            print(f"\n[{i}/{len(all_papers)}] {paper.title[:60]}...")
            print(f"   Category: {category.upper()}")
            
            # Download PDF
            filename = f"{arxiv_id}.pdf"
            filepath = data_dir / filename
            
            if filepath.exists():
                print(f"Already exists, skipping download...")
            else:
                paper.download_pdf(filename=str(filepath))
                print(f"Downloaded PDF")
            
            # Extract text
            print(f"Extracting text...")
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Store metadata
            metadata_list.append({
                'title': paper.title,
                'arxiv_id': arxiv_id,
                'category': category,
                'authors': paper_info['authors'],
                'published': paper_info['published'],
                'filepath': str(filepath),
                'num_pages': len(reader.pages)
            })
            
            # Store document
            processed_documents.append({
                'text': text,
                'title': paper.title,
                'arxiv_id': arxiv_id,
                'category': category,
                'authors': paper_info['authors'],
                'published': paper_info['published'],
                'num_pages': len(reader.pages),
                'num_chars': len(text)
            })
            
            print(f"Processed: {len(reader.pages)} pages, {len(text):,} chars")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n"+"="*60)
    print(f"   Successfully processed {len(processed_documents)} papers!")
    print(f"   Famous papers: {sum(1 for d in processed_documents if d['category'] == 'famous')}")
    print(f"   Recent papers: {sum(1 for d in processed_documents if d['category'] == 'recent')}")
    
    # Save metadata
    metadata_path = data_dir / "all_papers_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"\n Metadata saved to {metadata_path}")
    
    # Save processed documents
    processed_path = data_dir / "all_papers_processed.json"
    with open(processed_path, 'w') as f:
        json.dump(processed_documents, f, indent=2)
    print(f"💾 Processed documents saved to {processed_path}")
    
    return processed_documents

def main():
    """Main function to download everything"""
    print("="*60)
    print("Downloading famous ML papers + recent papers")
    print("="*60)
    
    # Get famous papers
    famous = download_famous_papers()
    print(f"\n{len(famous)} famous papers queued")
    
    # Get recent papers
    recent = download_recent_papers(num_papers=20)
    print(f"\n{len(recent)} recent papers queued")
    
    # Download and process all
    documents = download_and_process_all(famous, recent)
    
    print("\n"+"="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()