from config import Config
from openai import OpenAI
import time

client = OpenAI()
config = Config()
MODEL = config.openai.model


def embed_batch(texts, batch_size=100):
    """Embed a batch of texts"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=MODEL
        )
        embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(embeddings)
        time.sleep(0.1)  # Rate limiting
    return all_embeddings
    
    