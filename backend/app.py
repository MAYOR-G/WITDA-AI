# app.py
import os
import shutil
import datetime as dt
import logging
from typing import List, Optional, Literal, Dict, Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag import RAGPipeline, ensure_chroma
from .ai import AIClient
from .storage import ChatStore
from .sources import GITHUB_SOURCES, WEB_SOURCES

# ---- config ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DOCS_DIR = os.path.join(PROJECT_ROOT, "company_docs")
DB_PATH = os.path.join(PROJECT_ROOT, "memory.db")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_index")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---- services ----
ai = AIClient()  # Gemini models (generation + embeddings + scoring)
store = ChatStore(DB_PATH)
rag = RAGPipeline(
    docs_dir=DOCS_DIR,
    chroma_dir=CHROMA_DIR,
    ai=ai,
    store=store,  # [MODIFIED] Pass the store for chat history
    chunk_size=1200,
    chunk_overlap=220,
    k_retrieval=14,
    k_final=6
)

ensure_chroma(CHROMA_DIR)

# [MODIFIED] Add startup event to auto-index sources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    log.info("Server starting up...")
    log.info("Starting initial source refresh (GitHub & Web)...")
    try:
        stats = rag.refresh_seed_sources()
        log.info(f"Initial source refresh complete: {stats}")
    except Exception as e:
        log.error(f"Failed to refresh sources on startup: {e}")
    yield
    # On shutdown
    log.info("Server shutting down...")

app = FastAPI(
    title="WITDA-AI Backend", 
    version="1.2.0",
    lifespan=lifespan  # [MODIFIED] Use new lifespan manager
)

# Allow your local dev & deployed domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- models ----------------
Mode = Literal["default", "deep_research", "brief_explanation", "learn"]

class ChatIn(BaseModel):
    message: str
    mode: Mode = "default"
    chat_id: Optional[str] = None
    links: Optional[List[str]] = None
    persona: Optional[str] = "company_assistant"

class SourceOut(BaseModel):
    tag: str
    type: str
    name: str
    uri: str

class ChatOut(BaseModel):
    chat_id: str
    content: str
    citations: List[str] = []
    sources: List[SourceOut] = []

# ---------------- utilities ----------------
def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def _path(u: str) -> str:
    try:
        return urlparse(u).path.lower()
    except Exception:
        return ""

def infer_source_tag(uri: str, name: str) -> str:
    """
    Heuristic tagger for UI badges.
    Priority order tries to be specific first, general last.
    """
    # [MODIFIED] Handle local file paths more robustly
    if os.path.exists(uri) or ("company_docs" in uri) or (os.path.sep in uri and not uri.startswith("http")):
        if uri.lower().endswith(".pdf"):
            return "Paper"
        if uri.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "Image"
        return "Internal Doc"

    d = _domain(uri)
    p = _path(uri)

    if "github.com" in d:
        return "GitHub"
    if "arxiv.org" in d:
        return "Paper"
    if "research.google" in d or "ai.googleblog.com" in d:
        return "Blog"
    if "openai.com" in d:
        return "OpenAI"
    if "microsoft.com" in d and "/research" in p:
        return "Blog"
    if "kdnuggets.com" in d or "huggingface.co" in d:
        return "Blog"
        
    if name.lower().endswith(".pdf"):
        return "Paper"

    return "Web"

def attach_tags(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for s in sources:
        uri = s.get("uri") or ""
        name = s.get("name") or ""
        tag = infer_source_tag(uri, name)
        enriched.append({
            "tag": tag,
            "type": s.get("type", "web"),
            "name": name,
            "uri": uri
        })
    return enriched

# ---------------- health ----------------
@app.get("/health")
def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat()}

# ---------------- indexing ----------------
@app.post("/index/rebuild")
def rebuild_index():
    """Wipes and rebuilds index from ALL files in /company_docs"""
    added = rag.rebuild_all()
    return {"status": "ok", "added": added}

# ---------------- scraping ----------------
class ScrapeIn(BaseModel):
    urls: List[str]

@app.post("/scrape")
def scrape_and_index(req: ScrapeIn):
    """Scrapes and indexes a list of URLs."""
    added = rag.scrape_and_add(req.urls)
    return {"status": "ok", "added": added}

# ---------------- uploads ----------------
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and index PDF, TXT, MD, and Image files."""
    saved = []
    for f in files:
        # [MODIFIED] Sanitize filename
        safe_name = os.path.basename(f.filename or "unknown_file")
        dest_path = os.path.join(DOCS_DIR, safe_name)
        try:
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest_path)
        except Exception as e:
            log.error(f"Failed to save uploaded file {safe_name}: {e}")
            return {"status": "error", "message": f"Failed to save file: {safe_name}"}, 500
        finally:
            f.file.close()
            
    if not saved:
        return {"status": "error", "message": "No files were saved"}, 400
        
    n = rag.add_paths(saved)  # incremental index
    log.info(f"Uploaded and indexed {len(saved)} files, {n} chunks.")
    return {"status": "ok", "saved": [os.path.basename(s) for s in saved], "chunks_indexed": n}

# ---------------- sources ----------------
@app.get("/sources")
def list_sources():
    return {"github": GITHUB_SOURCES, "web": WEB_SOURCES}

@app.post("/sources/refresh")
def refresh_sources():
    """Crawl GitHub + scrape web sources declared in sources.py, then index."""
    log.info("Manual source refresh requested...")
    stats = rag.refresh_seed_sources()
    log.info(f"Manual source refresh complete: {stats}")
    return {"status": "ok", **stats}

# ---------------- chat ----------------
@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    # Optional: scrape links on-demand
    if req.links:
        try:
            rag.scrape_and_add(req.links)
        except Exception as e:
            log.error(f"Failed to scrape on-demand links: {e}")

    result = rag.answer(
        query=req.message.strip(),
        mode=req.mode,
        persona=req.persona,
        chat_id=req.chat_id
    )

    # Enrich sources with tags for UI badges
    tagged_sources = attach_tags(result.get("sources", []))

    # persist memory
    chat_id = result["chat_id"]
    store.save_message(
        chat_id=chat_id,
        role="user",
        content=req.message,
        meta={"mode": req.mode, "persona": req.persona}
    )
    store.save_message(
        chat_id=chat_id,
        role="assistant",
        content=result["content"],
        meta={"citations": result.get("citations", []), "sources": tagged_sources}
    )

    return ChatOut(
        chat_id=chat_id,
        content=result["content"],
        citations=result.get("citations", []),
        sources=[SourceOut(**s) for s in tagged_sources]
    )

# ---------------- history ----------------
@app.get("/history/{chat_id}")
def history(chat_id: str):
    return {"chat_id": chat_id, "messages": store.get_chat(chat_id)}