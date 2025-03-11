from config import Config
from openai import OpenAI
import time
from typing import List
client = OpenAI()
config = Config()
MODEL = config.openai.model


def embed_batch(texts: List[str], batch_size: int = 100, model: str = MODEL) -> List[List[float]]:
    """Embed a batch of texts"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(embeddings)
        time.sleep(0.1)  # Rate limiting
    return all_embeddings
    
    