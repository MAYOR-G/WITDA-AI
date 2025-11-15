# WITDA-AI - Advanced Retrieval-Augmented Generation (RAG) System

## Project Overview

WITDA-AI is a sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-rich answers to user queries. It achieves this by intelligently combining information from various sources: internal company documents, dynamically scraped web content, and publicly available GitHub repository data. Acting as an intelligent assistant, WITDA-AI grounds its responses in factual, relevant data, minimizing hallucinations and maximizing reliability.

The system is built with ease of deployment and configuration in mind, allowing users to seamlessly integrate their own knowledge sources and API keys for a highly tailored and powerful experience.

## Features

*   **Multi-Source Data Ingestion**: Ingests data from local files (PDFs, TXT, MD, CSV, Images), specified web URLs, and public GitHub repositories/organizations.
*   **Advanced Chunking Strategy**: Utilizes overlap chunking to preserve context across document splits, improving retrieval accuracy.
*   **LLM-Backed Query Expansion**: Expands user queries into multiple related variants using a Large Language Model (LLM) to overcome vocabulary mismatch and enhance retrieval.
*   **LLM-Based Reranking**: Employs an LLM to rerank initially retrieved document chunks, ensuring only the most relevant context is passed to the final generation model.
*   **Flexible Answer Generation Modes**: Supports various response styles:
    *   `Default`: Concise and conversational answers.
    *   `Deep Research`: Detailed, long-form research reports with structured sections.
    *   `Brief Explanation`: Comprehensive but concise briefings.
    *   `Learn`: In-depth tutorials for beginners, including examples.
*   **Conversational Memory**: Maintains chat history to provide context-aware responses in ongoing conversations.
*   **Dynamic Source Tagging**: Automatically tags sources (e.g., "Paper", "GitHub", "Internal Doc", "Web") for better organization and user understanding.
*   **Interactive Frontend**: A React-based web interface for seamless interaction, including:
    *   Chat history management.
    *   File upload functionality (PDFs, images, text files).
    *   On-demand URL scraping.
    *   Theme toggling (dark/light mode).
    *   Copy-to-clipboard for code blocks.
    *   Inline citation highlighting.
*   **Scalable Vector Storage**: Uses ChromaDB for efficient storage and retrieval of document embeddings.
*   **Google Gemini Integration**: Leverages Google Gemini models for embeddings, query expansion, reranking, and answer generation.

## Architecture and Data Flow

The WITDA-AI system follows a robust RAG architecture, comprising several interconnected components:

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
    *   **Local Documents**: Files from the `backend/company_docs` directory (e.g., PDFs, `.txt`, `.md`, images) are read and processed.
    *   **Web Scraper**: Configured URLs in `backend/sources.py` (or provided on-demand) are scraped for their textual content.
    *   **GitHub Ingester**: Public repositories and organizations specified in `backend/sources.py` are traversed, and allowed file types are fetched.
2.  **Text Chunking:** Ingested text is broken down into smaller, manageable chunks using an overlap strategy. This is crucial for efficient retrieval and to fit within the LLM's context window.
3.  **Embedding:** Each text chunk is converted into a numerical vector (embedding) using a specialized embedding model (Gemini's `text-embedding-004`). These embeddings capture the semantic meaning of the chunks.
4.  **Vector Database Storage:** The embeddings and their corresponding metadata (source, path, chunk index) are stored in a ChromaDB vector database for fast similarity search.
5.  **Query Expansion:** When a user submits a query, it's expanded into multiple related queries by an LLM to improve the chances of finding relevant documents, addressing the "vocabulary mismatch" problem.
6.  **Retrieval:** The expanded queries are embedded, and a similarity search is performed against the ChromaDB to retrieve the top `k` most relevant text chunks (candidates).
7.  **Reranking:** The retrieved candidate chunks are then reranked by an LLM to identify the most pertinent `k_final` chunks, ensuring the highest quality context for generation.
8.  **Generation:** The reranked context chunks, along with the original user query, a predefined persona, and conversational history, are fed into a large language model (Gemini's `gemini-2.5-flash`) to generate a coherent and informed answer.
9.  **Answer Formatting:** The generated answer is then formatted into a readable Markdown output, including inline citations to the original sources.

## Advanced RAG System: Chunking and Retrieval Strategies

### Overlap Chunking

The system employs **overlap chunking** as its primary text splitting strategy. Instead of simply splitting documents into discrete, non-overlapping chunks, each chunk shares a portion of text with the preceding and succeeding chunks.

**Why Overlap Chunking?**

*   **Context Preservation**: When an important piece of information spans across two chunk boundaries, overlap ensures that both chunks contain enough surrounding context. This significantly reduces the chance of losing critical information that might be crucial for accurate retrieval.
*   **Improved Retrieval**: By providing overlapping context, the embedding model has more information to create a rich and accurate vector representation for each chunk. This leads to better similarity matches during retrieval, even if the query aligns more with the "overlap" part of a chunk.
*   **Robustness to Split Points**: It makes the system less sensitive to arbitrary split points, as the context is preserved across these boundaries.

The `chunk_text` function in `rag.py` implements this with configurable `chunk_size` (default 1200 tokens) and `overlap` (default 220 tokens).

### Query Expansion

To enhance retrieval effectiveness, the system utilizes **LLM-backed query expansion**. This involves:

1.  Taking the initial user query.
2.  Using a large language model to generate several alternative phrasings, synonyms, or related terms for that query.
3.  Performing a retrieval query in the vector database with *all* these expanded queries.

This strategy helps overcome the "vocabulary mismatch" problem, where a user's query might use different terminology than the ingested documents, even if they refer to the same concept. By expanding the query, the system casts a wider net, increasing the likelihood of retrieving relevant documents.

### Reranking with LLM

After initial retrieval of candidate documents, the system employs **LLM-based reranking**. This is a critical step to refine the quality of the context provided to the final generation model:

1.  The initial retrieval fetches a larger set of potentially relevant chunks (`k_retrieval`, default 14).
2.  An LLM is then used to critically evaluate each of these candidate chunks against the original user query. The LLM assigns a relevance score to each chunk.
3.  Only the top `k_final` chunks (default 6) with the highest scores are selected as the final context for the answer generation.

**Benefits of LLM Reranking:**

*   **Improved Relevance**: LLMs are excellent at understanding nuanced semantic relationships, leading to a more precise selection of truly relevant chunks compared to simple vector similarity.
*   **Reduced Noise**: It filters out less relevant or redundant chunks that might have been picked up during the initial retrieval, providing a cleaner and more focused context.
*   **Better Answer Quality**: By feeding a highly refined set of context chunks, the final answer generation is more accurate, concise, and less prone to hallucination.

## Data Structures

The primary data structures managed by the system are:

*   **Text Chunks**: Strings of text derived from documents after chunking.
*   **Embeddings**: Numerical vector representations (lists of floats) of the text chunks, generated by the embedding model (`text-embedding-004`).
*   **Metadata**: A dictionary associated with each chunk and its embedding, containing information like:
    *   `source`: Original source filename or web URL (e.g., "Resume.pdf", "openai.com/news").
    *   `path`: Full path to the original document if local, or URI if scraped.
    *   `chunk`: Index of the chunk within its original document.
*   **ChromaDB Collection**: The persistent storage for embeddings and metadata, named `company_rag`.
*   **Chat History**: Stored in an SQLite database (`memory.db`) via `ChatStore` for conversational context.

## Function of Each File

### `backend/`

*   **`app.py`**: The FastAPI application entry point. It defines API endpoints for health checks, index rebuilding, URL scraping, file uploads, source refreshing, chat interactions, and chat history retrieval. It orchestrates the RAG pipeline and handles request/response serialization.
*   **`ai.py`**: Encapsulates all interactions with the Google Gemini models. This includes:
    *   Configuring the Gemini API key and models (`gemini-2.5-flash` for generation, `text-embedding-004` for embeddings).
    *   Generating text embeddings (`embed`) with appropriate task types (`retrieval_query` or `retrieval_document`).
    *   Performing text generation (`generate`) with system instructions and safety settings.
    *   Expanding user queries (`expand_query`).
    *   Reranking retrieved document candidates (`rerank`).
    *   Formatting the final answer (`format_answer`) based on the selected mode (default, deep\_research, brief\_explanation, learn).
*   **`rag.py`**: Contains the core RAG pipeline logic and data ingestion mechanisms. This includes:
    *   Document loading functions (`read_any`, `read_pdf`, `read_txt_like`, `read_image`, `scrape_url`, GitHub helpers).
    *   Text chunking (`chunk_text`) with configurable size and overlap.
    *   Interaction with ChromaDB (`add_paths`, `rebuild_all`).
    *   Orchestration of web and GitHub ingestion (`scrape_and_add`, `ingest_github_repo`, `ingest_github_owner`, `refresh_seed_sources`).
    *   The main `RAGPipeline` class which integrates these components for answering queries, managing retrieval, and generation.
*   **`sources.py`**: A configuration file for defining external data sources. It lists `GITHUB_SOURCES` (repositories/organizations) and `WEB_SOURCES` (URLs to scrape). It also specifies `GITHUB_ALLOWED_EXTS` for file filtering and safety limits for GitHub ingestion.
*   **`storage.py`**: Manages the SQLite database (`memory.db`) for persisting chat messages and history. It provides methods to `save_message` and `get_chat`.
*   **`memory.db`**: The SQLite database file used by `storage.py` to store conversational memory.
*   **`chroma_index/`**: Directory containing the persistent ChromaDB index files, which store the vector embeddings and metadata.
*   **`company_docs/`**: A directory intended for storing internal company documents (e.g., PDFs, resumes, project specifications) that the RAG system should reference.

### `frontend/`

*   **`index.html` / `in.html`**: These files (likely `in.html` is the primary one, `index.html` might be an older version or for specific deployment) contain the React-based single-page application (SPA) for the user interface. It handles:
    *   Displaying chat messages and history.
    *   User input via a text area and file/URL inputs.
    *   Selecting different answer generation modes.
    *   Uploading files to the backend.
    *   Displaying sources with dynamic tags and inline citations.
    *   Managing chat sessions (new chat, selecting existing chats).
    *   Theme toggling (dark/light mode).
    *   Copy-to-clipboard functionality for code blocks.

### Root Directory

*   **`main.py`**: The primary entry point for running the entire application. It typically sets up the backend server and serves the frontend.
*   **`pyproject.toml`**: Project configuration file, used for dependency management (e.g., with `uv` or `poetry`).
*   **`uv.lock`**: Lock file generated by the `uv` package manager, ensuring reproducible environments by pinning exact dependency versions.
*   **`requirements.txt`**: Lists Python dependencies required for the backend, including `fastapi`, `uvicorn`, `google-generativeai`, `chromadb`, `pypdf`, `beautifulsoup4`, `requests`, `Pillow`, and `pytesseract`.

## Implementation Guide

To implement and run the WITDA-AI RAG system, follow these steps:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd WITDA-AI
```

### 2. Set Up Your Python Environment

This project uses `uv` for dependency management. If you don't have it, install it:

```bash
pip install uv
```

Then, install the project dependencies:

```bash
uv sync
```

### 3. Install Tesseract OCR (for Image Uploads)

If you plan to upload image files (PNG, JPG, etc.) for OCR processing, you need to install Tesseract OCR on your system.

*   **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
*   **macOS**: `brew install tesseract`
*   **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

Ensure `pytesseract` is installed in your Python environment (it's in `backend/requirements.txt`).

### 4. Configure API Keys

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

### 5. Add Your Company Documents

Place all your relevant company documents (PDFs, `.txt`, `.md`, `.csv`, images, etc.) into the `WITDA-AI/backend/company_docs/` directory. These documents will be indexed and used by the RAG system.

### 6. Configure External Sources

Edit the `WITDA-AI/backend/sources.py` file to specify your desired external web and GitHub sources:

*   **`GITHUB_SOURCES`**: Add URLs of GitHub users, organizations, or specific repositories you want the system to ingest.
    *   Example: `"https://github.com/your-org-name"`, `"https://github.com/your-username/your-repo"`
*   **`WEB_SOURCES`**: Add URLs of websites, blogs, or research pages you want the system to scrape.
*   **`GITHUB_ALLOWED_EXTS`**: Modify the list if you need to include or exclude specific file extensions from GitHub repositories.

### 7. Rebuild the Knowledge Base

After adding new documents or modifying `sources.py`, you need to rebuild the RAG system's knowledge base. This process involves:

1.  Clearing the existing ChromaDB index.
2.  Reading all local documents from `company_docs`.
3.  Scraping all configured web sources.
4.  Ingesting content from all configured GitHub sources.
5.  Chunking, embedding, and indexing all this new content into ChromaDB.

This operation is automatically triggered on backend startup via the `lifespan` event in `app.py`. You can also manually trigger it via the `/index/rebuild` or `/sources/refresh` API endpoints (e.g., using `curl` or a UI button if available).

### 8. Run the Application

Execute the main application file to start the FastAPI backend server:

```bash
python main.py
# Or, for development with auto-reload:
uvicorn backend.app:app --reload --port 8000
```

### 9. Interact with the System

Open your browser to the frontend HTML file (e.g., `file:///path/to/WITDA-AI/frontend/in.html` or `http://localhost:8000/in.html` if served by a web server) and start asking questions!

### Things You Need to Edit/Change:

*   **API Keys**: `GEMINI_API_KEY` (mandatory) and `GITHUB_TOKEN` (optional but recommended for extensive GitHub use) as environment variables.
*   **Tesseract OCR**: Install on your system for image processing.
*   **Company Documents**: Populate `WITDA-AI/backend/company_docs/` with your internal knowledge.
*   **External Sources**: Update `WITDA-AI/backend/sources.py` to point to your desired web and GitHub content.
*   **Model Parameters (Advanced)**: For fine-tuning, you might adjust parameters like `chunk_size`, `chunk_overlap`, `k_retrieval`, `k_final` in `rag.py`, or generation parameters in `ai.py`. However, for most users, the defaults should be a good starting point.
