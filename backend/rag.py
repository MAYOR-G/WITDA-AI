# rag.py
import os
import re
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings

from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader 

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("WARNING: 'Pillow' or 'pytesseract' not installed. Image OCR will not work.")
    print("Install them with: pip install pillow pytesseract")
    Image = None
    pytesseract = None

from .storage import ChatStore
from .sources import (
    GITHUB_SOURCES, WEB_SOURCES,
    GITHUB_ALLOWED_EXTS, MAX_REPOS_PER_ORG, MAX_FILES_PER_REPO, MAX_BYTES_PER_FILE
)

log = logging.getLogger(__name__)

def ensure_chroma(path: str):
    os.makedirs(path, exist_ok=True)

# --------- utils ---------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    text = normalize_space(text)
    if not text:
        return []
    # Simple split, can be replaced with token-based chunking
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
    return [c for c in chunks if c] # Filter empty chunks

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_txt_like(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    out = []
    try:
        with open(path, "rb") as f:
            r = PdfReader(f)
            for p in r.pages:
                try:
                    out.append(p.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(out)
    except Exception as e:
        log.error(f"Failed to read PDF {path}: {e}")
        return ""

def read_image(path: str) -> str:
    if not Image or not pytesseract:
        log.warning(f"Skipping image file {path}, Pillow/pytesseract not installed.")
        return ""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        log.error(f"Failed to OCR image {path}: {e}")
        return ""

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".markdown", ".json", ".csv", ".rst", ".py", ".js", ".html", ".css"]:
        return read_txt_like(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return read_image(path)
    # Fallback for other unknown text-like files
    log.warning(f"Unknown file type {ext} for {path}, reading as text.")
    return read_txt_like(path)

def safe_filename(name: str, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:limit] or "page"

def scrape_url(url: str, timeout: int = 25) -> Tuple[str, str]:
    """Return (title, text)."""
    # [MODIFIED] Added User-Agent header to prevent 403 Forbidden errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    # drop scripts/styles/navs
    for tag in soup(["script","style","nav","header","footer","noscript","svg","form","aside"]):
        tag.decompose()
    text = normalize_space(soup.get_text(" ", strip=True))
    return title, text[:400_000]

# ----------------- GitHub helpers -----------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_API = "[https://api.github.com](https://api.github.com)" 

def gh_headers():
    # [MODIFIED] Added API version header, required by GitHub
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def is_repo_url(url: str) -> bool:
    return bool(re.match(r"^https://github\.com/[^/]+/[^/]+/?$", url))

def is_org_or_user_url(url: str) -> bool:
    return bool(re.match(r"^https://github\.com/[^/]+/?$", url)) and not is_repo_url(url)

def parse_owner_repo(url: str) -> Tuple[str, str] | None:
    m = re.match(r"^https://github\.com/([^/]+)/([^/]+)/?$", url)
    if not m: return None
    return m.group(1), m.group(2)

def list_repos_for_owner(owner: str, limit: int) -> List[Dict[str, Any]]:
    repos = []
    page = 1
    # [MODIFIED] Corrected API endpoint logic for user OR org
    url_user = f"{GITHUB_API}/users/{owner}/repos"
    url_org = f"{GITHUB_API}/orgs/{owner}/repos"
    
    while len(repos) < limit:
        # Try user endpoint first
        params = {"per_page": min(100, limit - len(repos)), "page": page, "type": "public", "sort": "updated"}
        r = requests.get(url_user, headers=gh_headers(), params=params)
        
        if r.status_code == 404:
            # If user not found, try org endpoint
            r = requests.get(url_org, headers=gh_headers(), params=params)
        
        r.raise_for_status() # Raise error if it's still not OK
        
        batch = r.json()
        if not batch: break
        repos.extend(batch)
        if len(batch) < 100: break
        page += 1
        
    return repos[:limit]

def list_repo_tree(owner: str, repo: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=gh_headers())
    r.raise_for_status()
    repo_meta = r.json()
    default_branch = repo_meta.get("default_branch", "main")
    
    r2 = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1", headers=gh_headers())
    r2.raise_for_status()
    data = r2.json()
    return data.get("tree", [])

def fetch_file_raw(owner: str, repo: str, path: str, max_bytes: int) -> str | None:
    # [MODIFIED] Use the correct raw content URL format
    url = f"[https://raw.githubusercontent.com/](https://raw.githubusercontent.com/){owner}/{repo}/HEAD/{path}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        
    r = requests.get(url, headers=headers, stream=True, timeout=25)
    if not r.ok: return None
    
    content = b""
    try:
        for chunk in r.iter_content(chunk_size=1024):
            content += chunk
            if len(content) > max_bytes:
                content = content[:max_bytes]
                break
    except Exception as e:
        log.error(f"Failed to stream file {url}: {e}")
        return None
        
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return None
        
# ----------------- RAG Pipeline -----------------
class RAGPipeline:
    def __init__(
        self,
        docs_dir: str,
        chroma_dir: str,
        ai: Any,
        store: ChatStore, 
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        k_retrieval: int = 12,
        k_final: int = 6,
    ):
        self.docs_dir = docs_dir
        self.ai = ai
        self.store = store 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.k_final = k_final

        self.client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(allow_reset=True))
        self.col = self.client.get_or_create_collection(
            name="company_rag",
            metadata={"hnsw:space": "cosine"}
        )

    # ------------- indexing -------------
    def add_paths(self, paths: List[str]) -> int:
        texts, metadatas, ids = [], [], []
        for p in paths:
            try:
                log.info(f"Reading: {p}")
                raw = read_any(p)
                if not raw or not raw.strip():
                    log.warning(f"Skipping empty file: {p}")
                    continue
                    
                filename = os.path.basename(p)
                for i, chunk in enumerate(chunk_text(raw, self.chunk_size, self.chunk_overlap)):
                    cid = hash_id(f"{p}::{i}")
                    texts.append(chunk)
                    metadatas.append({"source": filename, "path": p, "chunk": i})
                    ids.append(cid)
            except Exception as e:
                log.error(f"Failed to process path {p}: {e}")
                continue
        
        if texts:
            try:
                log.info(f"Embedding {len(texts)} chunks...")
                embs = self.ai.embed(texts)
                self.col.upsert(documents=texts, metadatas=metadatas, embeddings=embs, ids=ids)
                log.info(f"Upserted {len(texts)} chunks to Chroma.")
            except Exception as e:
                log.error(f"Failed to embed/upsert chunks: {e}")
                return 0
        return len(texts)

    def rebuild_all(self) -> int:
        log.info("Rebuilding index: resetting Chroma collection.")
        try:
            self.client.delete_collection("company_rag")
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(name="company_rag", metadata={"hnsw:space":"cosine"})
        
        added = 0
        paths_to_index = []
        for root, _, files in os.walk(self.docs_dir):
            for fn in files:
                if fn.endswith(".db"):
                    continue
                paths_to_index.append(os.path.join(root, fn))
        
        if paths_to_index:
            added = self.add_paths(paths_to_index)
        log.info(f"Rebuild complete. Added {added} chunks.")
        return added

    def scrape_and_add(self, urls: List[str]) -> int:
        paths = []
        for u in urls:
            try:
                log.info(f"Scraping: {u}")
                title, text = scrape_url(u)
                safe = safe_filename(title)
                scrape_dir = os.path.join(self.docs_dir, "scraped_web")
                os.makedirs(scrape_dir, exist_ok=True)
                
                path = os.path.join(scrape_dir, f"{safe}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{title}\n\nSource: {u}\n\n{text}")
                paths.append(path)
            except Exception as e:
                log.error(f"Failed to scrape URL {u}: {e}")
                continue
        return self.add_paths(paths)

    def ingest_github_repo(self, repo_url: str) -> Dict[str, int]:
        stats = {"files_saved": 0, "files_indexed": 0}
        pr = parse_owner_repo(repo_url)
        if not pr:
            log.error(f"Invalid repo URL: {repo_url}")
            return stats
            
        owner, repo = pr
        log.info(f"Ingesting GitHub repo: {owner}/{repo}")
        try:
            tree = list_repo_tree(owner, repo)
        except Exception as e:
            log.error(f"Failed to list repo tree for {owner}/{repo}: {e}")
            return stats
            
        saved_paths = []
        gh_dir = os.path.join(self.docs_dir, "github")
        os.makedirs(gh_dir, exist_ok=True)
        
        files_in_repo = 0
        for node in tree:
            if node.get("type") != "blob":
                continue
            
            files_in_repo += 1
            if files_in_repo > MAX_FILES_PER_REPO:
                log.warning(f"Hit max file limit for {owner}/{repo}, stopping.")
                break

            path = node.get("path", "")
            ext = os.path.splitext(path)[1].lower()
            
            if GITHUB_ALLOWED_EXTS and ext not in GITHUB_ALLOWED_EXTS:
                continue
            
            content = fetch_file_raw(owner, repo, path, MAX_BYTES_PER_FILE)
            if not content:
                continue
                
            safe = safe_filename(f"{owner}_{repo}_{path.replace('/', '_')}")
            local = os.path.join(gh_dir, f"{safe}.txt")
            
            with open(local, "w", encoding="utf-8") as f:
                f.write(f"Repo: {owner}/{repo}\nFile: {path}\nSource: [https://github.com/](https://github.com/){owner}/{repo}/blob/HEAD/{path}\n\n")
                f.write(content)
            saved_paths.append(local)
            stats["files_saved"] += 1
            
        stats["files_indexed"] = self.add_paths(saved_paths)
        log.info(f"Finished repo {owner}/{repo}: saved {stats['files_saved']}, indexed {stats['files_indexed']} chunks.")
        return stats

    def ingest_github_owner(self, owner_url: str) -> Dict[str, int]:
        stats = {"repos": 0, "files_saved": 0, "files_indexed": 0}
        owner = owner_url.rstrip("/").split("/")[-1]
        log.info(f"Ingesting GitHub owner: {owner}")
        try:
            repos = list_repos_for_owner(owner, limit=MAX_REPOS_PER_ORG)
        except Exception as e:
            log.error(f"Failed to list repos for {owner}: {e}")
            return stats
            
        for r in repos:
            repo_url = r.get("html_url")
            if not repo_url:
                continue
            sub = self.ingest_github_repo(repo_url)
            stats["repos"] += 1
            stats["files_saved"] += sub["files_saved"]
            stats["files_indexed"] += sub["files_indexed"]
        return stats

    def refresh_seed_sources(self) -> Dict[str, Any]:
        summary = {"web_saved": 0, "web_indexed": 0, "github": []}
        if WEB_SOURCES:
            try:
                indexed = self.scrape_and_add(WEB_SOURCES)
                summary["web_saved"] = len(WEB_SOURCES)
                summary["web_indexed"] = indexed
            except Exception as e:
                log.error(f"Failed to scrape WEB_SOURCES: {e}")
        for u in GITHUB_SOURCES:
            try:
                if is_repo_url(u):
                    g = self.ingest_github_repo(u)
                elif is_org_or_user_url(u):
                    g = self.ingest_github_owner(u)
                else:
                    g = {"error": "not_a_github_url"}
            except Exception as e:
                g = {"error": str(e)}
            summary["github"].append({"url": u, **g})
        return summary

    # ------------- retrieval -------------
    def _pre_retrieval(self, query: str) -> List[str]:
        expansions = self.ai.expand_query(query, k=6)[:6]
        all_q = [query] + expansions
        return list(dict.fromkeys(all_q)) # Deduplicate

    def _retrieve(self, queries: List[str], k: int) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            try:
                qv = self.ai.embed([q])[0]
                res = self.col.query(query_embeddings=[qv], n_results=k)
                for i in range(len(res["ids"][0])):
                    item = {
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i],
                        "meta": res["metadatas"][0][i],
                    }
                    seen[item["id"]] = item
            except Exception as e:
                log.error(f"Failed to retrieve for query '{q}': {e}")
                continue
        return list(seen.values())

    def _post_rerank(self, query: str, candidates: List[Dict[str, Any]], k_final: int) -> List[Dict[str, Any]]:
        return self.ai.rerank(query, candidates, top_k=k_final)

    # ------------- generation -------------
    def _build_context_block(self, items: List[Dict[str, Any]]) -> str:
        out = []
        for it in items:
            meta = it.get("meta", {}) or {}
            src = meta.get("source") or meta.get("path") or "Unknown Source"
            filename = os.path.basename(src)
            out.append(f"[{filename}] {it['text']}")
        return "\n\n".join(out[:12]) # Hard limit on context block

    def _gen(self, query: str, persona: str, context_items: List[Dict[str, Any]], mode: str, history: List[Dict[str, Any]]) -> str:
        context = self._build_context_block(context_items)
        
        history_block = ""
        if history:
            hist_msgs = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                hist_msgs.append(f"{role}: {msg['content']}")
            history_block = "\n".join(hist_msgs)

        sys = (
            "You are WITDA-AI, a helpful company assistant. "
            "You **always ground answers in provided context chunks** if available. "
            "If context is insufficient, state that briefly, then answer with best-effort general knowledge. "
            "You MUST cite sources from the context chunks like `(filename.pdf)`."
        )
        
        prompt = f"""Persona: {persona}

Previous conversation history (for context only):
\"\"\"\n{history_block}\n\"\"\"\n

Grounded context chunks (use these to answer, cite them as `(filename.ext)`):
\"\"\"\n{context}\n\"\"\"\n

Latest user query:
{query}

Rules:
- Synthesize context and history to answer the latest query.
- Never fabricate citations; only cite sources provided in the 'Grounded context chunks'.
- Use compact, copy-friendly Markdown.

Now, answer the latest user query based on the mode '{mode}'."""
        
        draft = self.ai.generate(sys, prompt)
        return self.ai.format_answer(mode, query, context, draft)

    # ------------- public -------------
    def answer(self, query: str, mode: str, persona: str, chat_id: str | None) -> Dict[str, Any]:
        
        # 1. Get history
        history = []
        current_chat_id = chat_id or str(uuid.uuid4())
        if chat_id:
            try:
                # Get last 6 messages (3 turns)
                history = self.store.get_chat(chat_id)[-6:]
            except Exception as e:
                log.error(f"Failed to get chat history for {chat_id}: {e}")
        
        # 2. Retrieval
        q_variants = self._pre_retrieval(query)
        candidates = self._retrieve(q_variants, k=self.k_retrieval)
        final_ctx = self._post_rerank(query, candidates, k_final=self.k_final)
        
        # 3. Generation
        content = self._gen(query, persona, final_ctx, mode, history)

        # 4. Collate sources
        cites: List[str] = []
        sources: List[Dict[str, Any]] = []
        for it in final_ctx:
            m = it.get("meta", {}) or {}
            uri = m.get("path") or m.get("source") or ""
            name = m.get("source") or os.path.basename(uri) or "Unknown"
            
            if uri and uri not in cites:
                cites.append(uri)
                sources.append({
                    "type": "doc" if (os.path.exists(uri) or "company_docs" in uri) else "web",
                    "name": name,
                    "uri": uri
                })

        return {
            "chat_id": current_chat_id,
            "content": content,
            "citations": cites,
            "sources": sources
        }