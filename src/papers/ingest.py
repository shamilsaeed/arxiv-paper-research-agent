import arxiv
import json
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import time

CATEGORIES = json.load(open("categories.json"))

class ArxivIngestor:
    def __init__(self):
        self.client = arxiv.Client(page_size= 2000, delay_seconds = 3, num_retries = 10)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/v1/paper/"
        self.categories = CATEGORIES
        self.processed_papers = 0
       
      # Can do 100 request per 5 min which is slow...i will expose this as  a tool for AI agent based on user needs
    # def get_citation_info(self, arxiv_id):
    #     url = f"{self.semantic_scholar_base_url}arXiv:{arxiv_id}"
    #     response = requests.get(url)
    #     if response.status_code == 200:
    #         data = response.json()
    #         return data.get('citationCount', 0), data.get('influentialCitationCount', 0), data.get('citationVelocity', 0)
    #     else:
    #         raise Exception(f"Failed to fetch citation info for {arxiv_id}")
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Remove version number from arxiv ID"""
        return arxiv_id.split('v')[0]

    def process_papers(self, year: int = 2024):
        """Fetch all Computer Science papers for given year"""
        print(f"Fetching CS papers from {year}")
        
        search = arxiv.Search(
            query=f"cat:cs.* AND submittedDate:[{year} TO {year+1}]",
            max_results=15000,  # Fetch all papers
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers_batch = {} # {category: [paper_data]}
        offset = 0
        page_position = 0  # Track position within current page

        while True:
            try:
                # Get results with current offset
                results = self.client.results(search, offset=offset)
                
                for i, paper in enumerate(results):
                    
                    if i < page_position:
                        continue
                    
                    try:
                        arxiv_id = paper.get_short_id()
                        paper_data = {
                            'title': paper.title,
                            'summary': paper.summary,
                            'arxiv_id': arxiv_id,
                            'category': paper.primary_category,
                            'published': paper.published.strftime('%Y-%m-%d'),
                            'pdf_url': paper.pdf_url
                        }
                        
                        category = paper.primary_category
                        if category not in papers_batch:
                            papers_batch[category] = []
                        papers_batch[category].append(paper_data)
                        
                        self.processed_papers += 1
                        print(f"Processed {self.processed_papers} papers")
                        page_position += 1
                        
                        if self.processed_papers % 100 == 0:
                            self._save_batch(papers_batch)
                            papers_batch = {}
                            
                    except Exception as e:
                        print(f"Error processing paper: {e}")
                        continue
                
                # Successfully processed this page, update offset
                offset += self.client.page_size
                page_position = 0  # Reset position for next page
            except arxiv.UnexpectedEmptyPageError:
                print(f"\nGot empty page at offset {offset}, sleeping for 10 seconds before retry...")
                time.sleep(10)  # Sleep longer to let API recover
                # Don't update offset - retry same page
                continue
                
            # except Exception as e:
            #     print(f"\nUnexpected error at offset {offset}: {e}")
            #     if papers_batch:
            #         self._save_batch(papers_batch)
            #     break


    def _save_batch(self, papers: Dict[str, List[Dict]]):
        """Save all accumulated papers when batch size is reached"""
        category_counts = {}
        
        for category, papers in papers.items():
            if papers:  # Only save categories that have papers
                category_name = self.categories.get(category, "unknown")
                output_file = self.data_dir / f"{category_name}_2024.jsonl"
                
                # Append mode
                with open(output_file, 'a') as f:
                    for paper in papers:
                        f.write(json.dumps(paper) + '\n')
                
                category_counts[category_name] = len(papers)
    
        # Single summary print
    #    summary = ", ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
    #    print(f"\nAppended papers to categories: {summary}")
        
        
if __name__ == "__main__":
    ingestor = ArxivIngestor()
    ingestor.process_papers(year=2024)