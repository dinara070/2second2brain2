"""Зашифроване сховище: SQLite в пам'яті + FTS5, на диску лише Fernet-шифртекст."""
from __future__ import annotations
import base64, datetime as dt, hashlib, os, re, sqlite3

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, source TEXT, title TEXT, body TEXT, ts REAL, hash TEXT);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id TEXT, idx INT, text TEXT, emb BLOB);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='unicode61 remove_diacritics 2');
"""


def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    out, cur = [], ""
    for p in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        while len(p) > size:
            if cur:
                out.append(cur); cur = ""
            out.append(p[:size]); p = p[size - overlap:]
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur); cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


class Vault:
    def __init__(self, path: str, passphrase: str):
        self.path = path
        salt_path = path + ".salt"
        if os.path.exists(salt_path):
            salt = open(salt_path, "rb").read()
        else:
            salt = os.urandom(16)
            open(salt_path, "wb").write(salt)
        key = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode())
        self.fernet = Fernet(base64.urlsafe_b64encode(key))
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        if os.path.exists(path):  # неправильний пароль -> InvalidToken
            self.db.deserialize(self.fernet.decrypt(open(path, "rb").read()))
        self.db.executescript(SCHEMA)

    def save(self) -> None:
        self.db.commit()  # serialize() не бачить незакомічених змін
        tmp = self.path + ".tmp"
        open(tmp, "wb").write(self.fernet.encrypt(self.db.serialize()))
        os.replace(tmp, self.path)

    def upsert(self, doc_id: str, source: str, title: str, body: str, ts: float, embedder=None) -> bool:
        h = hashlib.sha256((title + body).encode()).hexdigest()
        row = self.db.execute("SELECT hash FROM docs WHERE id=?", (doc_id,)).fetchone()
        if row and row[0] == h:
            return False
        if row:
            self._drop(doc_id)
        self.db.execute("INSERT INTO docs VALUES(?,?,?,?,?,?)", (doc_id, source, title, body, ts, h))
        date = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        header = f"[{source}] {title} ({date})"  # контекст у кожному фрагменті покращує пошук
        texts = [f"{header}\n{p}" for p in chunk(body) or [""]]
        embs = embedder.encode(texts) if embedder else [None] * len(texts)
        for i, (t, e) in enumerate(zip(texts, embs)):
            cur = self.db.execute("INSERT INTO chunks(doc_id,idx,text,emb) VALUES(?,?,?,?)",
                                  (doc_id, i, t, None if e is None else e.tobytes()))
            self.db.execute("INSERT INTO fts(rowid,text) VALUES(?,?)", (cur.lastrowid, t))
        return True

    def _drop(self, doc_id: str) -> None:
        for (cid,) in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall():
            self.db.execute("DELETE FROM fts WHERE rowid=?", (cid,))
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,))

    def stats(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT source, COUNT(*) FROM docs GROUP BY source"))

    def wipe(self) -> None:
        for p in (self.path, self.path + ".salt"):
            if os.path.exists(p):
                os.remove(p)
