import os
import json
import requests
import tempfile
from typing import Dict, List
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain.tools import Tool
from github import Github
from config import Config
from src.embed.vector_db import MilvusManager
from src.embed.embedder import embed_batch
config = Config()

CATEGORIES = json.load(open("categories.json"))

        # Actions:
        # - get_related_papers: Find relevant papers along with metadata
        # - get_summary: Get detailed paper summary
        # - get_citations: Get citations for a paper
        # - get_authors: Get author details and their other papers
        # - get_references: Get papers referenced by this paper
        # - get_metadata: Get publication date, categories, etc.
        # - get_github_repo: Get associated GitHub/code links if available
        # - get_citations: Get citations for a paper


class ResearchTools:
    def __init__(self, milvus_manager: MilvusManager):
        self.milvus = milvus_manager
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/v1/paper/"
        self.categories = CATEGORIES
        self.openai_client = OpenAI()
        self.arxiv_client = arxiv.Client()
        self.github_client = Github(config.github.token)
        self.models = {
            'embedding': "text-embedding-3-small",  # For vector search
            'summary': "gpt-4-turbo-preview",      # For detailed understanding
            'chat': "gpt-3.5-turbo"               # For simpler queries
        }
        
        self.tools = [
            Tool(
                name="get_related_papers",
                description="""Find research papers on a given topic along with metadata
                Input: Text query (e.g. "transformers in computer vision")
                Output: List of papers with titles, authors, publication dates, pdf_url, arxiv_id, and relevance scores""",
                func=self.get_related_papers
            ),
            Tool(
                name="get_paper_processed",
                description="""Process and store a paper in the vector database.
                Input: arxiv_id
                Output: Boolean (True if newly processed, False if already exists)""",
                func=self.get_paper_processed
            ),
            Tool(
                name="get_paper_details",
                description="""Answer specific questions about a processed paper using RAG.
                Only works if paper has been processed before.
                Input: Query string and arxiv_id
                Output: Detailed answer based on paper content""",
                func=self.get_paper_details
            ),
            Tool(
                name="get_summary",
                description="""Generate a structured summary of a paper.
                Input: arxiv_id
                Output: Comprehensive summary with problem, contributions, methodology, results, and conclusions""",
                func=self.get_summary
            ),
            Tool(
                name="get_citations",
                description="""Retrieve citation metrics for a paper.
                Input: arxiv_id
                Output: Citation statistics including count, influence, and velocity""",
                func=self.get_citations
            ),
            Tool(
                name="get_github_repo",
                description="""Find associated code repositories for a paper.
                Input: Paper title
                Output: List of relevant GitHub repositories""",
                func=self.get_github_repo
            )
        ]
    
    def get_github_repo(self, arxiv_title: str, k: int = 2) -> List[str]:
        """Get code repository of a paper."""
        
        query = f"{arxiv_title} implementation"
        repos = self.github_client.search_repositories(query=query, sort='stars', order='desc')
        valid_repos = list(repos)
        
        if valid_repos:
            return [x.full_name for x in valid_repos[:k]]
        else:
            return [] # found no repos

    def get_related_papers(self, query: str) -> List[Dict]:
        """
        Search for papers based on query and automatically identified categories.
        
        Input: String query like "transformers in computer vision"
        Returns: List of relevant papers
        """
        # Convert string input to expected format
        top_k = 5  # Default value

        # Get query embedding
        response = self.openai_client.embeddings.create(
            input=query,
            model=self.models['embedding']
        )
        query_embedding = response.data[0].embedding

        # Search in each identified category
        all_results = []

        results = self.milvus.search_similar(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format and add results
        for hits in results:
            for hit in hits:
                all_results.append({
                    'title': hit.entity.get('title'),
                    'arxiv_id': self._clean_arxiv_id(hit.entity.get('arxiv_id')),
                    'category': hit.entity.get('category'),
                    'published': hit.entity.get('published'),
                    'pdf_url': hit.entity.get('pdf_url'),
                    'authors': hit.entity.get('authors'),
                    'score': hit.distance
                })

        # 4. Sort by score and return top_k overall
        all_results.sort(key=lambda x: x['score'])
        return all_results[:top_k]

    def get_citations(self, arxiv_id: str) -> Dict:
        """Get citation metrics from Semantic Scholar API"""

        arxiv_id = self._clean_arxiv_id(arxiv_id)
        url = f"{self.semantic_scholar_base_url}arXiv:{arxiv_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'citation_count': data.get('citationCount', 0),
                'influential_citation_count': data.get('influentialCitationCount', 0),
                'citation_velocity': data.get('citationVelocity', 0)
            }
        else:
            raise Exception(f"Failed to fetch citation info for {arxiv_id}")
            
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean the arxiv_id to be used in the semantic scholar API"""
        return arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

    def _get_paper_collection_name(self, arxiv_id: str) -> str:
        """Get the collection name for a paper"""
        clean_id = arxiv_id.replace('.', '_') # Remove any quotes
        return f"paper_{clean_id}"
    
    def get_paper_processed(self, arxiv_id: str) -> Dict:
        """Check if paper has been processed before"""

        arxiv_id = self._clean_arxiv_id(arxiv_id)
        collection_name = self._get_paper_collection_name(arxiv_id)

        print(f"Checking if paper {collection_name} exists in Milvus")
        print(f"printing the arxiv_id: {arxiv_id}")
        # Check if paper is already processed
        if not self.milvus.has_collection(collection_name):
            print(f"Processing paper {collection_name} for the first time...")
                            
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.arxiv_client.results(search))
            
            # Download and process PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                paper.download_pdf(dirpath=os.path.dirname(temp_pdf.name),
                                 filename=os.path.basename(temp_pdf.name))
                temp_path = temp_pdf.name
            
            # Load and chunk PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(pages)

            # Prepare data for Milvus
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = embed_batch(chunk_texts)

            # Store in Milvus
            self.milvus.insert_paper_details(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings
            )
            print(f"Paper {arxiv_id} processed and stored in Milvus")
            return True
        
        else:
            print(f"Paper {arxiv_id} already exists in Milvus.")
            return False
        
    def get_summary(self, arxiv_id: str) -> Dict:
        """
        Get paper summary. If paper hasn't been processed before,
        download, chunk, and store it. If it exists, answer specific queries about it.
        
        Args:
            arxiv_id: The arxiv ID of the paper
            query: Optional specific question about the paper
        """
        arxiv_id = self._clean_arxiv_id(arxiv_id)
        collection_name = self._get_paper_collection_name(arxiv_id)
        
        # Get summary from Milvus
        all_chunks = self.milvus.get_all_paper_chunks(collection_name)  
        full_text = "\n".join(chunk['chunk_text'] for chunk in all_chunks)
        
        # Generate comprehensive summary
        summary_response = self.openai_client.chat.completions.create(
            model=self.models['summary'],
            messages=[
                {"role": "system", "content": """You are a research paper summarizer. 
                   Provide a comprehensive summary of the paper with:
                   1) Problem/Motivation
                   2) Key Contributions
                   3) Methodology
                   4) Main Results
                   5) Conclusions"""},
                {"role": "user", "content": full_text}
            ]
        )
        return {
            "arxiv_id": arxiv_id,
            "summary": summary_response.choices[0].message.content,
            "paper_processed": True
        }
        
    def get_paper_details(self, query: str, arxiv_id: str) -> Dict:
        """Get specific details about a paper in RAG format. Only works if paper has been processed before."""

        arxiv_id = self._clean_arxiv_id(arxiv_id)
        collection_name = self._get_paper_collection_name(arxiv_id)
        
        if not self.milvus.has_collection(collection_name):
            return {"error": f"Paper {arxiv_id} not found in database. Please process paper first."}
        
        # Get relevant chunks for query
        query_embedding = embed_batch([query])[0] 
        chunks = self.milvus.search_paper_chunks(
            query_embedding=query_embedding,
            arxiv_id=collection_name,
        )
        
        # Generate answer
        context = "\n".join(chunk['chunk_text'] for chunk in chunks)
        
        answer_response = self.openai_client.chat.completions.create(
                model=self.models['chat'],
                messages=[
                    {"role": "system", "content": """You are an expert at analyzing research papers. 
                    Answer questions about the paper using only the provided context. 
                    Be precise and technical in your response. 
                    If the context doesn't contain enough information to answer the question, say so.
                    Do not make up information, only use the context provided."""},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
        
        return {
            "arxiv_id": arxiv_id,
            "query": query,
            "answer": answer_response.choices[0].message.content
        }