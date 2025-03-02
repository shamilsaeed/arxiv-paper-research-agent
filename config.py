import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class MilvusConfig:
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", 19530))
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
      
@dataclass
class ArxivConfig:
    max_results: int = 100
    batch_size: int = 10
    categories: list = None  # if None, will fetch all CS categories
    rate_limit: float = 1.0  # seconds between requests
    
@dataclass
class Config:
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    

# Example usage:
if __name__ == "__main__":
    config = Config()
    print(f"Milvus host: {config.milvus.milvus_host}")
    print(f"Milvus port: {config.milvus.milvus_port}")
#    print(f"OpenAI model: {config.openai.model}")
#    print(f"Arxiv batch size: {config.arxiv.batch_size}")