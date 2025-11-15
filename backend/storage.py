# storage.py
import sqlite3
import json
import os
import datetime as dt
from typing import Any, Dict, List

class ChatStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure()

    def _ensure(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            ts TEXT,
            role TEXT,
            content TEXT,
            meta TEXT
        )""")
        conn.commit(); conn.close()

    def save_message(self, chat_id: str, role: str, content: str, meta: Dict[str, Any] | None = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages(chat_id, ts, role, content, meta) VALUES (?, ?, ?, ?, ?)",
            (chat_id, dt.datetime.utcnow().isoformat(), role, content, json.dumps(meta or {}))
        )
        conn.commit(); conn.close()

    def get_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT ts, role, content, meta FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
        rows = c.fetchall(); conn.close()
        out = []
        for ts, role, content, meta in rows:
            try: meta = json.loads(meta) if meta else {}
            except Exception: meta = {}
            out.append({"ts": ts, "role": role, "content": content, "meta": meta})
        return out
