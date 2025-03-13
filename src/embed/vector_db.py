from typing import Any, Dict, List

from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

from config import Config

config = Config()
MILVUS_HOST = config.milvus.milvus_host
MILVUS_PORT = config.milvus.milvus_port
PAPERS_COLLECTION = config.milvus.milvus_collection


class MilvusManager:
    def __init__(self):
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        self._connect()
        self._init_collection()
        
    def _connect(self) -> None:
        """Connect to Milvus"""
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print("Connected to Milvus")
        
    def _init_collection(self) -> None:
        """Initialize the papers collection"""
        if not utility.has_collection(PAPERS_COLLECTION):
            self.create_collection()
            print(f"Created new paper collection: {PAPERS_COLLECTION}")
        else:
            print(f"Collection already exists: {PAPERS_COLLECTION}")
        # Load the collection
        collection = Collection(PAPERS_COLLECTION)
        collection.load()
        print(f"Loaded collection: {PAPERS_COLLECTION}")
        
        
    def create_collection(self) -> Collection:
        """Create the papers collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="published", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="pdf_url", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim) 
        ]
        schema = CollectionSchema(fields, description="ArXiv papers collection")
        collection = Collection(name=PAPERS_COLLECTION, schema=schema)
        
        collection.create_index("embedding", self.index_params, index_name="papers_index")
        return collection
    
    
    def _process_authors(self, authors_str: str, max_authors: int = 20) -> str:
        """Process authors string to include at most max_authors"""
        # Split authors by comma and clean
        authors_list = [author.strip() for author in authors_str.split(',')]
        
        # Take only up to max_authors
        truncated_authors = authors_list[:max_authors]
        
        # Add indicator if we truncated the list
        if len(authors_list) > max_authors:
            truncated_authors.append(f"... and {len(authors_list) - max_authors} more")
            
        return ', '.join(truncated_authors)

    def insert_papers(self, papers: List[Dict[str, Any]]) -> None:
        """Insert papers into the collection"""
        collection = Collection(PAPERS_COLLECTION)
        
        # Check if any papers already exist - deduplicate
        results = collection.query(
            expr="arxiv_id != ''",
            output_fields=["arxiv_id"]
        )
        existing_arxivids = {r["arxiv_id"] for r in results}
        papers = [p for p in papers if p['arxiv_id'] not in existing_arxivids]
        
        if not papers:
            print("No new papers to insert")
            return
        
        # Process papers to limit authors
        processed_papers = []
        for paper in papers:
            paper['authors'] = self._process_authors(paper['authors'])
            processed_papers.append(paper)
        
        # Prepare data for insertion
        entities = [
            [p['title'] for p in processed_papers],
            [p['arxiv_id'] for p in processed_papers],
            [p['category'] for p in processed_papers],
            [p['published'] for p in processed_papers],
            [p['pdf_url'] for p in processed_papers],
            [p['authors'] for p in processed_papers],
            [p['embedding'] for p in processed_papers]
        ]
        collection.insert(entities)
        print(f"Inserted {len(processed_papers)} papers into collection")
        
    def search_similar_papers(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar papers"""
        collection = Collection(PAPERS_COLLECTION)
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
            output_fields=["title", "arxiv_id", "category", "published", "pdf_url", "authors"]
        )

        # Format results for easier consumption
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'title': hit.entity.get('title'),
                    'arxiv_id': hit.entity.get('arxiv_id'),
                    'category': hit.entity.get('category'),
                    'published': hit.entity.get('published'),
                    'pdf_url': hit.entity.get('pdf_url'),
                    'authors': hit.entity.get('authors'),
                    'score': hit.distance
                })
        
        return formatted_results
    
    def search_paper_chunks(self, query_embedding: List[float], arxiv_id: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks within a specific paper"""
        collection_name = arxiv_id  # The collection name is already properly formatted when passed in
        
        if not utility.has_collection(collection_name):
            raise ValueError(f"{collection_name} not found in milvus database")
        
        collection = Collection(collection_name)
        collection.load()
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
            output_fields=["chunk_text", "chunk_id"]
        )

        # Format results for easier consumption
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'chunk_text': hit.entity.get('chunk_text'),
                    'chunk_id': hit.entity.get('chunk_id'),
                    'score': hit.distance
                })
        
        # Sort by chunk_id to maintain document context
        formatted_results.sort(key=lambda x: x['chunk_id'])
        return formatted_results
    
    def get_collection(self) -> Collection:
        """Get the papers collection"""
        return Collection(PAPERS_COLLECTION)
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        return utility.has_collection(collection_name)
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete the papers collection"""
        utility.drop_collection(collection_name)
        print(f"Dropped collection: {collection_name}")

    def insert_paper_details(self, collection_name: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Create a new collection for requested paper and insert its chunks"""

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, description=f"Paper chunks for {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        
        collection.create_index("embedding", self.index_params)
        print(f"Created new collection for paper: {collection_name}")
        
        # Get collection and insert chunks
        collection = Collection(collection_name)
        collection.load()
        
        # Prepare data for insertion
        entities = [
            [chunk.page_content for chunk in chunks],  # chunk texts
            embeddings  # embeddings
        ]
        
        collection.insert(entities)
        print(f"Inserted {len(chunks)} chunks for paper {collection_name}")
        
    def get_all_paper_chunks(self, collection_name: str) -> List[Dict]:
        """Retrieve all chunks for a paper in order"""
        
        collection = Collection(collection_name)
        collection.load()
        
        # Query all chunks
        results = collection.query(
            expr="chunk_id >= 0",  # Get all documents
            output_fields=["chunk_text"],
        )
        # Sort chunks by chunk_id if available
        sorted_results = sorted(results, key=lambda x: x["chunk_id"])
        return sorted_results
    
    def search_similar(self,  query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar papers in a category"""
        
        collection = Collection(PAPERS_COLLECTION)
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
            output_fields=["title", "arxiv_id", "category", "published", "pdf_url"]
        )       
        
        return results 