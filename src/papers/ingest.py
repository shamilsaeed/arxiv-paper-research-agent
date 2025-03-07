import arxiv
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

CATEGORIES = json.load(open("categories.json"))

class ArxivIngestor:
    def __init__(self):
        self.client = arxiv.Client(page_size= 1000, num_retries = 3)
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

    def process_papers(self, start_date: str, end_date: str):
        """Fetch all Computer Science papers for given date range
        Args:
            start_date: Format 'YYYYMM'
            end_date: Format 'YYYYMM'
        """
        print(f"Fetching CS papers from {start_date} to {end_date}")
        
        # Create month-year directory
        month_year_dir = self.data_dir / f"{start_date}"
        month_year_dir.mkdir(exist_ok=True)
        self.current_data_dir = month_year_dir
               
        search = arxiv.Search(
            query=f"cat:cs.* AND submittedDate:[{start_date} TO {end_date}]",
            max_results=None,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers_batch = {} # {category: [paper_data]}

        while True:
            try:
                # Get results with current offset
                results = self.client.results(search)
                
                for paper in results:
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
                        
                        if self.processed_papers % 100 == 0:
                            self._save_batch(papers_batch)
                            papers_batch = {}
                            
                    except Exception as e:
                        print(f"Error processing paper: {e}")
                        continue
                
                if papers_batch:
                    self._save_batch(papers_batch)
                    
            except arxiv.UnexpectedEmptyPageError:
                print(f"\nGot empty page, sleeping for few seconds before retry...")
                noise = np.random.normal(0, 10) 
                sleep_time = 5 + noise 
                time.sleep(sleep_time) 
                # Don't update offset - retry same page
                continue
                

    def _save_batch(self, papers: Dict[str, List[Dict]]):
        """Save all accumulated papers when batch size is reached"""
        category_counts = {}
        
        for category, papers in papers.items():
            if papers:  # Only save categories that have papers
                category_name = self.categories.get(category, "unknown")
                output_file = self.current_data_dir / f"{category_name}.jsonl"
                
                # Append mode
                with open(output_file, 'a') as f:
                    for paper in papers:
                        f.write(json.dumps(paper) + '\n')
                
                category_counts[category_name] = len(papers)
    
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Ingest ArXiv papers for a specific date range')
    parser.add_argument('--start-date', type=str, required=True,
                      help='Start date in YYYY-MM format (e.g., 2024-01)')
    parser.add_argument('--end-date', type=str, required=True,
                      help='End date in YYYY-MM format (e.g., 2024-02)')
    
    args = parser.parse_args()
    
    ingestor = ArxivIngestor()
    ingestor.process_papers(args.start_date, args.end_date)