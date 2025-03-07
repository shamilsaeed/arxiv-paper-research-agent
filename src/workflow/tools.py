import os
import json
import requests
import tempfile
from typing import Dict, List
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from langchain.tools import Tool
from github import Github
from config import Config
from src.embed.vector_db import MilvusManager

config = Config()

CATEGORIES = json.load(open("categories.json"))

        # Actions:
        # - get_related_papers: Find relevant papers
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
                description="""Search for research papers. Use when user wants to find papers about a topic.
                Input: A simple text query string (e.g. "transformers in computer vision")
                Returns: list of relevant papers with titles, summaries, and scores.""",
                func=self.get_related_papers
            ),
            Tool(
                name="get_citations",
                description="""Get citation metrics and impact data for a paper.
                Use when user asks about:
                - paper influence
                - citation counts
                - research impact
                Input: arxiv_id
                Returns: citation statistics""",
                func=self.get_citations
            ),
            Tool(
                name="get_summary",
                description="""Generate detailed summary of a paper.
                Use when user wants to:
                - understand paper contents
                - get key findings
                - know methodology
                Input: arxiv_id
                Returns: structured summary""",
                func=self.get_summary
            ),
            Tool(
                name="get_authors",
                description="""Get authors of a paper.
                Use when user wants to know the authors of a paper.
                Input: arxiv_id
                Returns: list of authors""",
                func=self.get_authors
            ),
            Tool(
                name="get_github_repo",
                description="""Get code repository of a paper.
                Use when user wants to see the code of a paper.
                Input: arxiv_id
                Returns: list of authors""",
                func=self.get_github_repo
            )
        ]
        
    def get_authors(self, arxiv_id: str) -> List[str]:
        """Get authors of a paper."""
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(self.arxiv_client.results(search))
        return [a.name for a in paper.authors]
    
    def get_github_repo(self, arxiv_title: str, top_k: int = 2) -> List[str]:
        """Get code repository of a paper."""
        query = f"{arxiv_title} implementation"
        repositories = self.github_client.search_repositories(query=query, sort='stars', order='desc')
        return [x.repo for x in repositories[:top_k]]


    def get_related_papers(self, query: str) -> List[Dict]:
        """
        Search for papers based on query and automatically identified categories.
        
        Input: String query like "transformers in computer vision"
        Returns: List of relevant papers
        """
        # Convert string input to expected format
        top_k = 5  # Default value

        # 1. Identify relevant categories from query
        categories = self._identify_categories(query)
        print(f"Identified categories: {categories}")

        # 2. Get query embedding
        response = self.openai_client.embeddings.create(
            input=query,
            model=self.models['embedding']
        )
        query_embedding = response.data[0].embedding

        # 3. Search in each identified category
        all_results = []
        for category in categories:
            results = self.milvus.search_similar(
                category=category,
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
                        'score': hit.distance
                    })

        # 4. Sort by score and return top_k overall
        all_results.sort(key=lambda x: x['score'])
        return all_results[:top_k]

    def get_citations(self, arxiv_id: str) -> Dict:
        """Get citation metrics from Semantic Scholar API"""

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
        
    def get_summary(self, arxiv_id: str) -> Dict:
        """
        Summarize a research paper from its arxiv_id.
        Returns structured summary excluding references.
        """
        try:
            # 1. Download PDF to temp file
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.arxiv_client.results(search))

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                # Download directly to temp file
                paper.download_pdf(dirpath=os.path.dirname(temp_pdf.name),
                                 filename=os.path.basename(temp_pdf.name))
                temp_path = temp_pdf.name
            
            # 2. Load and process PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            text = "\n".join(page.page_content for page in pages)
            
            # Direct summarization with GPT-4 (better for longer texts)
            response = self.openai_client.chat.completions.create(
                model=self.models['summary'],  # Handles longer context
                messages=[
                    {"role": "system", "content": "Summarize this research paper with: 1) Key Findings 2) Methodology 3) Results"},
                    {"role": "user", "content": text}
                ]
            )
            
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            return {"error": str(e)}
        
        return {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "summary": response.choices[0].message.content
        }
                
    def _identify_categories(self, query: str) -> List[str]:
        """
        Identify relevant ArXiv categories from a natural language query.
        Returns list of category IDs (e.g., ['cs.AI', 'cs.LG'])
        """
        try:
            # Define category descriptions for better matching

            prompt = f"""Given this research query: "{query}"

            And these ArXiv CS categories:
            {self.categories}

            Return a Python list containing the most relevant category IDs (maximum 3).
            Only include the category values (e.g., artificial_intelligence, distributed_computing).
            If unsure, default to ['unknown'].

            Response format: ['category1', 'category2', 'category3']
            """

            response = self.openai_client.chat.completions.create(
                model=self.models['chat'],  # Using cheaper model for simple classification
                messages=[
                    {"role": "system", "content": "You are a research paper classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Want consistent categorization
            )
            
            # Parse the list directly
            try:
                # Safely evaluate the string as a list
                categories = eval(response.choices[0].message.content)
                if not isinstance(categories, list):
                    raise ValueError("Response not in list format")
            except:
                # Fallback if eval fails
                categories = ['unknown']
            
            print(f"Identified categories: {categories}")
            return categories
        
        except Exception as e:
            print(f"Error in category identification: {e}")
            return ['unknown']
        
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean the arxiv_id to be used in the semantic scholar API"""
        return arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
