# ArXiv Paper Research Assistant

A sophisticated research assistant (ReAct) that helps users find, process, and understand academic papers using vector search and large language models

## 🌟 Features

- **Smart Paper Search**: Find relevant academic papers based on natural language queries
- **Deep Paper Analysis**: Process and analyze PDF papers using RAG (Retrieval Augmented Generation)
- **Interactive Q&A**: Ask specific questions about papers and get contextual answers
- **Citation Metrics**: Track paper impact through citation statistics
- **Code Discovery**: Find associated GitHub repositories for papers
- **Vector Search**: Efficient similarity search using Milvus vector database
- **Memory Management**: Maintains conversation context for better interactions

## 🏗️ Architecture

### Core Components

- **Vector Database**: Milvus for efficient similarity search and paper storage
- **Language Models**: 
  - Groq for chat interactions
  - OpenAI for embeddings and summarization
- **Document Processing**: PDF parsing and chunking for detailed analysis
- **API Integrations**:
  - arXiv for paper access
  - Semantic Scholar for citation metrics
  - GitHub for code repository discovery

### Key Classes

- `ResearchAssistant`: Main agent class handling user interactions
- `ResearchTools`: Collection of tools for paper analysis
- `MilvusManager`: Vector database operations manager

## 🛠️ Technical Details

### Vector Collections

- **Main Collection**: Stores paper metadata and embeddings using paper summary
- **Paper Collections**: Individual collections for each processed paper's chunks


## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   # Install dependencies
   poetry install
   ```

2. **Configuration**
   - Create a `config.py` file with your API keys:
     - Groq API key
     - OpenAI API key
     - GitHub token
   - Configure Milvus connection settings
   - Run docker compose to start Milvus

3. **Running the Assistant**
   ```python
   from src.workflow.agent import ResearchAssistant
   from src.embed.vector_db import MilvusManager
   from src.workflow.tools import ResearchTools

   # Initialize components
   milvus_manager = MilvusManager()
   research_tools = ResearchTools(milvus_manager)
   agent = ResearchAssistant(research_tools)
   ```