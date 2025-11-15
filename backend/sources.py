# sources.py
"""
Central place to register always-on sources.
Add/modify URLs and repos here, no code changes required elsewhere.
"""

# GitHub orgs/repos to ingest (public). You can put org ONLY or specific repos.
# Examples:
#   "https://github.com/xvengm"              (org/user — will traverse public repos)
#   "https://github.com/xvengm/HealthBox"    (single repo)
GITHUB_SOURCES = [
    "https://github.com/xvengm",
]

# High-signal news/research/blog sources to keep fresh.
# (We scrape their HTML and store as .txt in company_docs/ then index.)
WEB_SOURCES = [
    "https://openai.com/research/",
    "https://openai.com/news/",
    "https://arxiv.org/list/cs.AI/recent",
    "https://ai.googleblog.com/",
    "https://www.microsoft.com/en-us/research/blog/",
    "https://www.kdnuggets.com/",
    "https://huggingface.co/blog",
    "https://rss.feedspot.com/ai_rss_feeds/",
    "https://onlinedegrees.sandiego.edu/ai-blogs/",
    "https://dev.to/ai_jobsnet/a-huge-list-of-aiml-news-sources-31n8",
]

# Optional: file extensions you’d like to ingest from GitHub
GITHUB_ALLOWED_EXTS = [
    ".md", ".rst", ".txt",
    ".py", ".ipynb", ".json", ".yml", ".yaml",
    ".js", ".ts", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c", ".cs",
    ".sql", ".ini", ".toml",
]

# Safety limits to avoid rate limit explosions
MAX_REPOS_PER_ORG = 25
MAX_FILES_PER_REPO = 400
MAX_BYTES_PER_FILE = 800_000  # ~0.8MB text per file
