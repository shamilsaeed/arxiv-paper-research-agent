# ArXiv Paper Research Assistant

A sophisticated research assistant (ReAct Agent) that helps users find, process, and understand academic papers using vector search and large language models

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

### 1. Environment Setup
```bash
# Install poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### 2. Configuration
- Create a `config.py` file with your API keys:
  - Groq API key
  - OpenAI API key
  - GitHub token
- Configure Milvus connection settings

### 3. Start Milvus
```bash
# Start Milvus using Docker Compose
docker compose up -d
```

### 4. Data Pipeline Setup

1. **Ingest Papers**
   ```bash
   # Ingest papers from ArXiv within a date range. e.g:
   poetry run python src/papers/ingest.py --start-date 20240301 --end-date 20240330
   ```

2. **Load Papers to Milvus**
   ```bash
   # Process and load ingested papers into Milvus
   poetry run python src/embed/main.py
   ```

### 5. Run the Application
```bash
# Start the research assistant
streamlit run src/app.py
```

## 💬 Usage Examples

Once the application is running, you can:

1. **Search for Papers**
   ```
   You: Find papers about transformer models in computer vision
   ```

2. **Process a Paper**
   ```
   You: Process paper 2103.14030
   ```

3. **Ask Questions**
   ```
   You: What were the main results of the paper?
   ```

4. **Get Citations**
   ```
   You: How many citations does this paper have?
   ```

## 📝 System Requirements

- Python 3.11+
- Docker and Docker Compose
- 16GB RAM recommended
- Sufficient disk space for paper storage

## 🔍 Troubleshooting

- If Milvus connection fails, ensure Docker containers are running:
  ```bash
  docker ps
  ```

## 🤝 Improvements

- Need to speed up the processing of papers as it is heavy operation
- Need to improve the agent to parse inputs better
- Need to add more error handling
- Need to improve the vector database to be more efficient
