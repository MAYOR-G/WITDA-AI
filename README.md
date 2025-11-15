# WITDA - Advanced Retrieval-Augmented Generation (RAG) System

## Project Overview

WITDA is an advanced Retrieval-Augmented Generation (RAG) system designed to answer user queries by leveraging a combination of internal company documents, scraped web content, and publicly available GitHub repository information. It acts as an intelligent assistant, grounding its responses in factual, relevant data to provide accurate and context-rich answers.

The system is built to be easily deployable and configurable, allowing users to integrate their own knowledge sources and API keys for a tailored experience.

## Architecture and Data Flow

The WITDA system follows a typical RAG architecture, comprising several key components:

```
+-------------------+     +-------------------+     +-------------------+
|  User Query       | --> |  Query Expansion  | --> |  Vector DB (ChromaDB) |
|  (Frontend)       |     |  (AIClient)       |     |  (RAGPipeline)    |
+-------------------+     +-------------------+     +-------------------+
          |                                                   ^
          |                                                   |  Retrieval
          V                                                   |
+-------------------+     +-------------------+     +-------------------+
|  LLM (Generative) | <-- |  Reranking        | <-- |  Context Chunks   |
|  (AIClient)       |     |  (AIClient)       |     |                   |
+-------------------+     +-------------------+     +-------------------+
          |
          V
+-------------------+
|  Formatted Answer |
|  (Frontend)       |
+-------------------+

Data Ingestion Flow:
+-------------------+     +-------------------+     +-------------------+
|  Local Documents  | --> |  Document Loader  | --> |  Text Chunker     |
|  (PDF, TXT, etc.) |     |  (rag.py)         |     |  (rag.py)         |
+-------------------+     +-------------------+     +-------------------+
          |                                                   |
+-------------------+     +-------------------+     +-------------------+
|  Web Scraper      | --> |  Web Content      | --> |  Embedding Model  |
|  (rag.py)         |     |                   |     |  (AIClient)       |
+-------------------+     +-------------------+     +-------------------+
          |                                                   |
+-------------------+     +-------------------+     +-------------------+
|  GitHub Ingester  | --> |  GitHub Content   | --> |  Vector DB (ChromaDB) |
|  (rag.py)         |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
```

### Key Stages:

1.  **Data Ingestion:**
    *   **Local Documents**: Files from the `backend/company_docs` directory (e.g., PDFs, `.txt` files) are read and processed.
    *   **Web Scraper**: Configured URLs in `backend/sources.py` are scraped for their textual content.
    *   **GitHub Ingester**: Public repositories and organizations specified in `backend/sources.py` are traversed, and allowed file types are fetched.
2.  **Text Chunking:** Ingested text is broken down into smaller, manageable chunks. This is crucial for efficient retrieval and to fit within the LLM's context window.
3.  **Embedding:** Each text chunk is converted into a numerical vector (embedding) using a specialized embedding model (Gemini's `text-embedding-004`). These embeddings capture the semantic meaning of the chunks.
4.  **Vector Database Storage:** The embeddings and their corresponding metadata (source, path, chunk index) are stored in a ChromaDB vector database.
5.  **Query Expansion:** When a user submits a query, it's expanded into multiple related queries by an LLM to improve the chances of finding relevant documents.
6.  **Retrieval:** The expanded queries are embedded, and a similarity search is performed against the ChromaDB to retrieve the top `k` most relevant text chunks (candidates).
7.  **Reranking:** The retrieved candidate chunks are then reranked by an LLM to identify the most pertinent `k_final` chunks, ensuring the highest quality context for generation.
8.  **Generation:** The reranked context chunks, along with the original user query and a predefined persona, are fed into a large language model (Gemini's `gemini-2.5-flash`) to generate a coherent and informed answer.
9.  **Answer Formatting:** The generated answer is then formatted into a readable Markdown output, including citations to the original sources.

## Advanced RAG System: Chunking and Retrieval Strategies

### Overlap Chunking

The system employs **overlap chunking** as its primary text splitting strategy. Instead of simply splitting documents into discrete, non-overlapping chunks, each chunk shares a portion of text with the preceding and succeeding chunks.

**Why Overlap Chunking?**

*   **Context Preservation**: When an important piece of information spans across two chunk boundaries, overlap ensures that both chunks contain enough surrounding context. This significantly reduces the chance of losing critical information that might be crucial for accurate retrieval.
*   **Improved Retrieval**: By providing overlapping context, the embedding model has more information to create a rich and accurate vector representation for each chunk. This leads to better similarity matches during retrieval, even if the query aligns more with the "overlap" part of a chunk.
*   **Robustness to Split Points**: It makes the system less sensitive to arbitrary split points, as the context is preserved across these boundaries.

The `chunk_text` function in `rag.py` implements this with configurable `chunk_size` (default 1200 tokens) and `overlap` (default 200 tokens).

### Query Expansion

To enhance retrieval effectiveness, the system utilizes **LLM-backed query expansion**. This involves:

1.  Taking the initial user query.
2.  Using a large language model to generate several alternative phrasings, synonyms, or related terms for that query.
3.  Performing a retrieval query in the vector database with *all* these expanded queries.

This strategy helps overcome the "vocabulary mismatch" problem, where a user's query might use different terminology than the ingested documents, even if they refer to the same concept. By expanding the query, the system casts a wider net, increasing the likelihood of retrieving relevant documents.

### Reranking with LLM

After initial retrieval of candidate documents, the system employs **LLM-based reranking**. This is a critical step to refine the quality of the context provided to the final generation model:

1.  The initial retrieval fetches a larger set of potentially relevant chunks (`k_retrieval`).
2.  An LLM is then used to critically evaluate each of these candidate chunks against the original user query. The LLM assigns a relevance score to each chunk.
3.  Only the top `k_final` chunks with the highest scores are selected as the final context for the answer generation.

**Benefits of LLM Reranking:**

*   **Improved Relevance**: LLMs are excellent at understanding nuanced semantic relationships, leading to a more precise selection of truly relevant chunks compared to simple vector similarity.
*   **Reduced Noise**: It filters out less relevant or redundant chunks that might have been picked up during the initial retrieval, providing a cleaner and more focused context.
*   **Better Answer Quality**: By feeding a highly refined set of context chunks, the final answer generation is more accurate, concise, and less prone to hallucination.

## Data Structures

The primary data structures managed by the system are:

*   **Text Chunks**: Strings of text derived from documents after chunking.
*   **Embeddings**: Numerical vector representations (lists of floats) of the text chunks, generated by the embedding model.
*   **Metadata**: A dictionary associated with each chunk and its embedding, containing information like:
    *   `source`: Original source filename or web URL (e.g., "Resume.pdf", "openai.com/news").
    *   `path`: Full path to the original document if local, or URI if scraped.
    *   `chunk`: Index of the chunk within its original document.
*   **ChromaDB Collection**: The persistent storage for embeddings and metadata.

## Function of Each File

### `backend/`

*   **`app.py`**: (Presumed) The main entry point for the backend API, handling HTTP requests and orchestrating the RAG pipeline.
*   **`ai.py`**: Handles all interactions with the Google Gemini models. This includes:
    *   Configuring the Gemini API key and models.
    *   Generating text embeddings (`embed`).
    *   Performing text generation (`generate`).
    *   Expanding user queries (`expand_query`).
    *   Reranking retrieved document candidates (`rerank`).
    *   Formatting the final answer (`format_answer`).
*   **`rag.py`**: Contains the core RAG pipeline logic. This includes:
    *   Document loading functions (`read_any`, `read_pdf`, `read_txt_like`, `scrape_url`, GitHub helpers).
    *   Text chunking (`chunk_text`).
    *   Interaction with ChromaDB (`add_paths`, `rebuild_all`).
    *   Orchestration of web and GitHub ingestion (`scrape_and_add`, `ingest_github_repo`, `ingest_github_owner`, `refresh_seed_sources`).
    *   The main `RAGPipeline` class which integrates all these components for answering queries.
*   **`sources.py`**: Configuration file for defining external data sources, including:
    *   `GITHUB_SOURCES`: List of GitHub repositories or organizations to ingest.
    *   `WEB_SOURCES`: List of URLs to scrape for web content.
    *   `GITHUB_ALLOWED_EXTS`: Whitelist of file extensions for GitHub ingestion.
    *   Safety limits for GitHub ingestion.
*   **`storage.py`**: (Not yet reviewed, but likely handles persistent storage beyond ChromaDB, possibly for chat history or other session data).
*   **`memory.db`**: (Presumed) SQLite database for storing conversational memory or other application-specific data.
*   **`chrom-index/`**: Directory containing the persistent ChromaDB index files (`chroma.sqlite3`).
*   **`company_docs/`**: Directory for storing internal company documents (e.g., PDFs, resumes, project specifications) that the RAG system should reference.
*   **`requirements.txt`**: Lists Python dependencies required for the backend.

### `frontend/`

*   **`in.html`**: (Presumed) The main HTML file for the frontend user interface, responsible for sending user queries and displaying answers.

### Root Directory

*   **`main.py`**: (Presumed) Entry point for running the entire application, potentially setting up the backend server and serving the frontend.
*   **`pyproject.toml`**: Project configuration file, often used with `poetry` or `pipenv` for dependency management.
*   **`uv.lock`**: Lock file for `uv` package manager, ensuring reproducible environments.

## Implementation Guide

To implement and run the WITDA RAG system, follow these steps:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd witda
```

### 2. Set Up Your Environment

This project uses `uv` for dependency management. If you don't have it, install it:

```bash
pip install uv
```

Then, install the project dependencies:

```bash
uv sync
```

### 3. Configure API Keys

You **MUST** set your Google Gemini API key as an environment variable. Without this, the AI components will not function.

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# For Windows (PowerShell):
# $env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

You can obtain a Gemini API key from the [Google AI Studio](https://ai.google.dev/).

If you plan to ingest from private GitHub repositories or to avoid GitHub API rate limits, set your GitHub Personal Access Token:

```bash
export GITHUB_TOKEN="YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
# For Windows (PowerShell):
# $env:GITHUB_TOKEN="YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
```
You can generate a GitHub Personal Access Token with `repo` scope in your [GitHub Developer Settings](https://github.com/settings/tokens).

### 4. Add Your Company Documents

Place all your relevant company documents (PDFs, `.txt`, `.md`, `.csv`, etc.) into the `witda/backend/company_docs/` directory. These documents will be indexed and used by the RAG system.

### 5. Configure External Sources

Edit the `witda/backend/sources.py` file to specify your desired external web and GitHub sources:

*   **`GITHUB_SOURCES`**: Add URLs of GitHub users, organizations, or specific repositories you want the system to ingest.
    *   Example: `"https://github.com/your-org-name"`, `"https://github.com/your-username/your-repo"`
*   **`WEB_SOURCES`**: Add URLs of websites, blogs, or research pages you want the system to scrape.
*   **`GITHUB_ALLOWED_EXTS`**: Modify the list if you need to include or exclude specific file extensions from GitHub repositories.

### 6. Rebuild the Knowledge Base

After adding new documents or modifying `sources.py`, you need to rebuild the RAG system's knowledge base. This process involves:

1.  Clearing the existing ChromaDB index.
2.  Reading all local documents.
3.  Scraping all configured web sources.
4.  Ingesting content from all configured GitHub sources.
5.  Chunking, embedding, and indexing all this new content into ChromaDB.

This operation can be triggered (presumably) via a backend endpoint or a specific script. You will need to interact with the `RAGPipeline` class, likely through `main.py` or a dedicated script, to call `rebuild_all()` and `refresh_seed_sources()`.

### 7. Run the Application

Execute the main application file:

```bash
python main.py
```

This will typically start the backend server and make the frontend accessible (likely via `in.html`).

### 8. Interact with the System

Open your browser to the address provided by the application (e.g., `http://localhost:8000/in.html`) and start asking questions!

### Things You Need to Edit/Change:

*   **API Keys**: `GEMINI_API_KEY` (mandatory) and `GITHUB_TOKEN` (optional but recommended for extensive GitHub use) as environment variables.
*   **Company Documents**: Populate `witda/backend/company_docs/` with your internal knowledge.
*   **External Sources**: Update `witda/backend/sources.py` to point to your desired web and GitHub content.
*   **Model Parameters (Advanced)**: For fine-tuning, you might adjust parameters like `chunk_size`, `chunk_overlap`, `k_retrieval`, `k_final` in `rag.py`, or generation parameters in `ai.py`. However, for most users, the defaults should be a good starting point.
