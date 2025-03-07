import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any
from config import Config

CATEGORIES = json.load(open("categories.json"))
config = Config()
MILVUS_HOST = config.milvus.milvus_host
MILVUS_PORT = config.milvus.milvus_port


class MilvusManager:
    def __init__(self,  categories: Dict[str, str]):
        self.categories = categories
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self._connect()
        self._create_collections()
        
    def _connect(self) -> None:
        """Connect to Milvus"""
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print("Connected to Milvus")
        
    def _create_collections(self) -> None:
        """Create collections for all categories"""
        categories_values = list(self.categories.values())
        for category in categories_values:
            if not utility.has_collection(category):
                self.create_collection(category)
            #    print(f"Created new collection: {category}")
            #else:
            #    print(f"Collection {category} already exists. Skipping...")
                
            # Load the collection
            collection = Collection(category)
            collection.load()
        print(f"Loaded collections!")
        
    def create_collection(self, category: str) -> Collection:
        """Create a new collection for a Arxiv category"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="published", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="pdf_url", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim) 
        ]
        schema = CollectionSchema(fields, description=f"ArXiv papers in {category}")
        collection = Collection(name=category, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params, index_name=f"{category}_index")
        return collection
    
    def get_or_create_collection(self, category: str) -> Collection:
        """Get existing collection or create new one"""
        if not utility.has_collection(category):
            return self.create_collection(category)
        return Collection(category)
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        return utility.list_collections()
    
    def drop_collection(self, category: str) -> None:
        """Drop a collection"""
        if utility.has_collection(category):
            utility.drop_collection(category)
            print(f"Dropped collection {category}")
            
    def insert_papers(self, category: str, papers: List[Dict[str, Any]]) -> None:
        """Insert papers into a collection"""
        collection = self.get_or_create_collection(category)
        
        # Check if any papers already exist - deduplicate
        results = collection.query(
            expr="arxiv_id != ''",  # or simply "1" to match all
            output_fields=["arxiv_id"]
        )
        existing_arxivids = {r["arxiv_id"] for r in results}
        papers = [p for p in papers if p['arxiv_id'] not in existing_arxivids]
        
        # Prepare data for insertion
        entities = [
            [p['title'] for p in papers],
            [p['arxiv_id'] for p in papers],
            [p['category'] for p in papers],
            [p['published'] for p in papers],
            [p['pdf_url'] for p in papers],
            [p['embedding'] for p in papers]
        ]
        
        collection.insert(entities)
        print(f"Inserted {len(papers)} papers into {category} collection")
        
    def search_similar(self, category: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar papers in a category"""
        collection = Collection(category)
        collection.load()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "arxiv_id", "category", "published", "pdf_url"]
        )       
        
        return results 
    