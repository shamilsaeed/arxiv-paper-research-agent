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
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "arxiv_papers")
    milvus_paper_details_collection: str = os.getenv("MILVUS_PAPER_DETAILS_COLLECTION", "arxiv_paper_details")
      
@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY")
    model: str = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    
@dataclass
class GithubConfig:
    token: str = os.getenv("GITHUB_TOKEN")
    
@dataclass
class GroqConfig:
    api_key: str = os.getenv("GROQ_API_KEY")
    model: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

@dataclass
class Config:
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    github: GithubConfig = field(default_factory=GithubConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)


if __name__ == "__main__":
    config = Config()
    print(f"Milvus host: {config.milvus.milvus_host}")
    print(f"Milvus port: {config.milvus.milvus_port}")
