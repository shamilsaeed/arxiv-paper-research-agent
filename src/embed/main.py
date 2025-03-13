import json
from pathlib import Path

from tqdm import tqdm

from config import Config
from src.embed.embedder import embed_batch
from src.embed.vector_db import MilvusManager
from src.papers.ingest import CATEGORIES

CATEGORIES = json.load(open("categories.json"))


if __name__ == "__main__":
    
    # Load config
    config = Config()
    categories = CATEGORIES

    # Initialize vector database
    vector_db = MilvusManager()

    # List collections
    print(vector_db.get_collection())
    
    # Insert papers
    data_dir = Path("data/raw")
    jsonl_files = list(data_dir.glob("**/*.jsonl"))
    
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        print(f"\nProcessing {jsonl_file}")
        papers = []
        summaries = []
        category = jsonl_file.stem

        # Read papers and collect summaries
        with open(jsonl_file, 'r') as f:
            for line in f:
                paper = json.loads(line.strip())
                papers.append(paper)
                summaries.append(paper['summary'])
        
        if papers:
            # Get embeddings
            print(f"Embedding {len(papers)} papers...")
            embeddings = embed_batch(summaries)
            
            # Add embeddings to papers
            for paper, embedding in zip(papers, embeddings):
                paper['embedding'] = embedding
            
            # Insert into Milvus
            print("Inserting into Milvus...")
            vector_db.insert_papers(papers)