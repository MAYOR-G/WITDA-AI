# ai.py
import os
import json
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    # Some SDK versions expose different enum names; we guard their import.
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except Exception:
    HarmCategory = None
    HarmBlockThreshold = None

# -------- config --------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR API KEY").strip()
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash").strip()
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-004").strip()

__all__ = ["AIClient"]


# -------- helpers --------
def _build_safety_settings() -> Optional[list]:
    """
    Returns a list of safety settings supported by the installed SDK.
    Works across versions where enum names differ or are missing.
    """
    if HarmCategory is None or HarmBlockThreshold is None:
        return None

    possible_names = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUAL_CONTENT",     # some versions
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",  # others
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    settings = []
    for name in possible_names:
        cat = getattr(HarmCategory, name, None)
        if cat is not None:
            settings.append({"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE})
    return settings or None


def _safe_extract_text(res) -> str:
    """Safely extract text from a Gemini response object."""
    try:
        if hasattr(res, "text") and res.text:
            return (res.text or "").strip()
        # stitch candidate parts if present
        if getattr(res, "candidates", None):
            for cand in res.candidates:
                parts = getattr(getattr(cand, "content", None), "parts", None)
                if parts:
                    out = " ".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "")]).strip()
                    if out:
                        return out
    except Exception:
        pass
    return ""


def _flatten_embedding_obj(obj) -> List[float]:
    """
    Accepts any of the SDK's shapes and always returns a flat List[float].
    """
    if isinstance(obj, dict):
        if "embeddings" in obj:  # batch shape
            return obj["embeddings"][0]["values"]
        if "embedding" in obj:   # single shape
            e = obj["embedding"]
            return e["values"] if isinstance(e, dict) else e
        if "values" in obj:      # raw
            return obj["values"]
    if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
        return obj
    raise RuntimeError(f"Unexpected embedding shape from SDK: {type(obj)}")


# -------- client --------
class AIClient:
    """
    Wrapper around Google Gemini:
      - embed(texts)
      - generate(system, prompt)
      - expand_query(q, k)
      - rerank(query, candidates, top_k)
      - format_answer(mode, query, context, draft)
    """

    def __init__(self):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. `pip install google-generativeai`")
        if not GEMINI_API_KEY:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")
        genai.configure(api_key=GEMINI_API_KEY)

    # ---------- embeddings ----------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Always return: one flat vector per input string (List[List[float]]).
        We call the API per-item to avoid SDK shape quirks.
        """
        if not isinstance(texts, list):
            texts = [texts]
        vecs: List[List[float]] = []
        for t in texts:
            # [MODIFIED] Switched to 'retrieval_query' or 'retrieval_document' task_type for clarity
            # Use 'retrieval_document' for indexing, 'retrieval_query' for searching
            task = "retrieval_document" if len(t) > 200 else "retrieval_query"
            out = genai.embed_content(model=EMB_MODEL, content=t, task_type=task)
            vecs.append(_flatten_embedding_obj(out))
        return vecs

    # ---------- text generation ----------
    def generate(self, system: str, prompt: str) -> str:
        """
        Gemini 1.5+ compatible:
          - pass system via system_instruction
          - build safety settings dynamically across SDK versions
          - safe text extraction (no crash if empty)
        """
        safety = _build_safety_settings()
        try:
            model = genai.GenerativeModel(
                GEN_MODEL,
                system_instruction=system,
                safety_settings=safety,
                generation_config={
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "top_k": 40,
                    # [MODIFIED] Increased token limit for detailed responses
                    "max_output_tokens": 8192, 
                },
            )
        except Exception:
            # If safety settings cause issues on this SDK, retry without them
            model = genai.GenerativeModel(
                GEN_MODEL,
                system_instruction=system,
                generation_config={
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
            )

        # IMPORTANT: role must be "user" (no "model") to avoid 400 invalid role errors.
        res = model.generate_content([{"role": "user", "parts": [prompt]}])
        return _safe_extract_text(res)

    # ---------- query expansion ----------
    def expand_query(self, q: str, k: int = 6) -> List[str]:
        """
        LLM-backed query expansion with safe fallback.
        Returns up to k short expansions (synonyms, acronyms, related terms).
        """
        sys = (
            "You expand search queries for a retrieval system. "
            "Return ONLY a bullet list (one per line), no numbering, no prose, max 12 items. "
            "Prefer short noun phrases and common variants; include acronyms if relevant."
        )
        prompt = f"Base query:\n{q}\n\nReturn distinct expansions. Keep each <= 6 words."
        txt = self.generate(sys, prompt) or ""
        # Parse bullet lines
        lines = [l.strip("-•* \t").strip() for l in txt.splitlines() if l.strip()]
        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for l in lines:
            low = l.lower()
            if low and low not in seen:
                seen.add(low)
                out.append(l)
            if len(out) >= k:
                break

        if not out:  # Fallback heuristics
            out.append(q) # Always include the original query
            
        return list(dict.fromkeys(out))[:k] # Deduplicate and limit

    # ---------- rerank ----------
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Try LLM-based JSON-lines scoring; if empty/unparsable, fallback to keyword overlap heuristic.
        """
        if not candidates:
            return []
            
        sys = (
            "You are a passage ranker. Score each chunk for usefulness in answering the query. "
            "Output JSON lines ONLY in the form: {\"idx\": <int>, \"score\": <0..1>}."
        )
        items = "\n".join([f"[{i}] {c['text'][:1000]}" for i, c in enumerate(candidates)])
        prompt = f"Query: {query}\n\nChunks:\n{items}\n\nOne JSON line per chunk."

        txt = self.generate(sys, prompt)
        scores: Dict[int, float] = {}
        if txt:
            for line in txt.splitlines():
                line = line.strip()
                if not line.startswith("{") or not line.endswith("}"):
                    continue
                try:
                    obj = json.loads(line)
                    idx = int(obj["idx"])
                    score = float(obj["score"])
                    if 0.0 <= score <= 1.0 and idx < len(candidates):
                        scores[idx] = score
                except Exception:
                    continue

        # Fallback: keyword overlap if LLM gave nothing usable
        if not scores:
            q_terms = {t for t in query.lower().split() if len(t) > 2}
            for i, c in enumerate(candidates):
                text = c["text"].lower()
                overlap = sum(1 for t in q_terms if t in text)
                scores[i] = float(overlap / (len(q_terms) + 1)) # Basic normalization

        ranked: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            c2 = dict(c)
            c2["score"] = scores.get(i, 0.0)
            ranked.append(c2)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    # ---------- answer formatter ----------
    def format_answer(self, mode: str, query: str, context: str, draft: str) -> str:
        # [MODIFIED] Centralized system prompt
        sys = (
            "You are a helpful and professional research assistant (WITDA-AI). "
            "Your job is to generate clear, detailed, and well-structured reports in Markdown. "
            "You MUST synthesize information from the 'Grounded context' (if provided) and your own general knowledge. "
            "Always output clean, readable Markdown. "
            "**CRITICAL RULE:** When you use a fact from the 'Grounded context', you MUST cite its source in parentheses at the end of the sentence, like `(Source Name)`. Example: 'The sky is blue (internal_report.pdf)'. "
            "Do NOT invent citations. Only cite sources from the context."
        )

        style = ""

        # [MODIFIED] Improved 'default' prompt
        if mode == "default":
            style = """## Task: Standard Answer
You are a helpful assistant. Please answer the user's query clearly and concisely.
Your primary goal is to be helpful, accurate, and conversational.

1.  **Check Context First:** Analyze the 'Grounded context'. If it directly answers the query, synthesize it and provide the answer. **Cite your sources** like `(source_name.pdf)`.
2.  **General Knowledge Fallback:** If the context is empty, not relevant, or insufficient, state that and then answer the query using your general knowledge, just like a standard large language model.
3.  **Structure:** Format the answer using clear Markdown. Use headings, bullet points, or numbered lists as needed.
4.  **Code:** If the user asks for code, provide it in a fenced code block with the language specified (e.g., ```python)."""

        # [MODIFIED] Massively improved 'deep_research' prompt
        elif mode == "deep_research":
            style = (
                "## Task: Deep Research Report\n"
                "You are an expert research analyst. Your job is to produce a detailed, long-form, and well-structured research report (1000-3000 words) in clean Markdown. "
                "You must synthesize BOTH the 'Grounded context' and your own extensive general knowledge.\n\n"
                "### REQUIRED OUTPUT STRUCTURE (Do NOT skip sections):\n\n"
                "## 1. Executive Summary\n"
                "A concise summary (2-4 sentences) of the key findings and answer.\n\n"
                "## 2. Detailed Analysis\n"
                "This is the main body. Provide a comprehensive, in-depth explanation. "
                "Use your general knowledge to expand on the topic, define terms, and provide background. "
                "Use sub-headings (e.g., `### 2.1 Sub-Topic`) to structure your analysis. "
                "**This section must be very detailed.**\n\n"
                "## 3. Context-Grounded Insights\n"
                "In this section, explicitly list the key facts or insights derived *specifically* from the 'Grounded context'. "
                "Use a bulleted list. **Cite each insight** like: `- Insight A (source_file.pdf)`. "
                "If no context was provided or relevant, state 'No specific insights were drawn from the provided context.'\n\n"
                "## 4. Code or Practical Examples (If Applicable)\n"
                "If the query involves code, methodology, or steps, provide clear, commented examples in fenced code blocks (e.g., ```python). If not applicable, state 'No practical examples are applicable for this query.'\n\n"
                "## 5. Conclusion & Next Steps\n"
                "Summarize the main takeaways and, if appropriate, suggest next steps or further reading.\n\n"
                "### RULES\n"
                "- **Be Thorough:** This is a *deep* research report. The answer MUST be detailed and educational. Do not provide a short answer.\n"
                "- **Cite Your Context:** You MUST follow the citation rule: `(source_name.pdf)`."
            )

        # [MODIFIED] Improved 'brief_explanation' prompt
        elif mode == "brief_explanation":
            style = (
                "## Task: Brief Explanation\n"
                "Please provide a comprehensive but concise briefing (approx. 300-500 words) on the user's query. "
                "You MUST generate full content for each section, not just an outline. "
                "Use the following Markdown structure:\n\n"
                "## 1. Introduction\n"
                "Define the core topic and its purpose (2-3 sentences).\n\n"
                "## 2. Key Concepts\n"
                "Explain the 3-5 main ideas in a bulleted list. Provide a 1-2 sentence explanation for each bullet.\n\n"
                "## 3. Practical Applications & Significance\n"
                "Describe why this topic is important with 2-3 concrete examples.\n\n"
                "## 4. Summary\n"
                "Recap the 3-4 most important takeaways as a bulleted list."
            )
        
        # [MODIFIED] Improved 'learn' prompt (to prevent incomplete responses)
        else:  # 'learn' mode
            style = (
                "## Task: Detailed Tutorial\n"
                "You are an expert teacher. Produce an in-depth tutorial (approx. 500-1000 words) for someone learning this topic from scratch. "
                "You MUST generate full, detailed content for every single section. Do not skip any. Do not just write headings.\n\n"
                "### REQUIRED OUTPUT STRUCTURE:\n\n"
                "## Introduction\n"
                "Provide a 2-3 paragraph overview defining the topic, the problem it solves, and why a beginner should learn it.\n\n"
                "## Key Concepts\n"
                "Use a bulleted list for at least 3-5 core concepts. For EACH concept:\n"
                "- **Concept Name:** Provide a detailed explanation (2-4 sentences).\n"
                "- **Example:** Give a short, concrete example or analogy to make it clear.\n\n"
                "## Why It Matters & Applications\n"
                "Use a numbered list to describe the practical importance and 3-5 key real-world applications. Explain *why* it's used there.\n\n"
                "## Walkthrough / Code Example\n"
                "Provide a compact, step-by-step example. If code is relevant, provide a complete, runnable code block (e.g., ```python) with comments explaining each line. If not, use a clear step-by-step list.\n\n"
                "## Summary & Takeaways\n"
                "Conclude with 3-5 key bullet points summarizing what the learner should remember."
            )

        p = f"""User query: {query}

Grounded context (verbatim snippets may appear):
\"\"\"\n{context}\n\"\"\"\n

Initial draft (use as a starting point, but apply new structure):
\"\"\"\n{draft}\n\"\"\"\n

Apply the requested structure for mode '{mode}' EXACTLY. Fill all sections with detailed content.
REMEMBER THE CITATION RULE: Cite sources like `(source_name.pdf)`."""
        
        out = self.generate(sys, f"{style}\n\n{p}")
        
        # Fallback if generation fails
        if out:
            return out
        elif draft:
            return f"**Fallback (unable to apply style):**\n\n{draft}"
        else:
            return "I apologize, but I was unable to generate a response for this query."
