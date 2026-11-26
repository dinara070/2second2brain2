"""Гібридний пошук: BM25 (FTS5) + семантика, злиття через RRF, фільтр за датою з тексту питання."""
from __future__ import annotations
import datetime as dt, re
import numpy as np

STOP = {"що", "як", "коли", "для", "про", "від", "який", "яка", "які", "між", "ми", "чи", "мені", "було", "був", "ось", "цей"}
MONTHS = {"січ": 1, "лют": 2, "берез": 3, "квіт": 4, "трав": 5, "черв": 6, "лип": 7,
          "серп": 8, "вересн": 9, "жовт": 10, "листопад": 11, "груд": 12}


class Embedder:
    def __init__(self, name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)  # працює локально, дані нікуди не йдуть

    def encode(self, texts, query: bool = False):
        pre = "query: " if query else "passage: "
        return self.m.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=32).astype("float32")


def load_embedder():
    try:
        return Embedder()
    except Exception:
        return None


def parse_dates(q: str, now: dt.datetime | None = None):
    """'у березні', 'в березні 2025', 'вчора', 'минулого тижня' -> (start_ts, end_ts) або None."""
    now, ql = now or dt.datetime.now(), q.lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "вчора" in ql:
        return (today - dt.timedelta(days=1)).timestamp(), today.timestamp()
    if "минулого тижня" in ql or "минулому тижні" in ql:
        mon = today - dt.timedelta(days=today.weekday())
        return (mon - dt.timedelta(days=7)).timestamp(), mon.timestamp()
    if "цього тижня" in ql:
        return (today - dt.timedelta(days=today.weekday())).timestamp(), (now + dt.timedelta(days=1)).timestamp()
    for stem, m in MONTHS.items():
        if re.search(rf"\b{stem}\w{{0,4}}\b", ql):
            y = re.search(r"\b(20\d\d)\b", ql)
            year = int(y.group(1)) if y else (now.year if m <= now.month else now.year - 1)
            start = dt.datetime(year, m, 1)
            end = dt.datetime(year + (m == 12), m % 12 + 1, 1)
            return start.timestamp(), end.timestamp()
    return None


def fts_query(q: str) -> str:
    terms = []
    for w in re.findall(r"\w+", q.lower()):
        if w in STOP or len(w) < 3:
            continue
        stem = w[:-2] if len(w) > 6 else w[:-1] if len(w) > 4 else w  # груба «стемінізація» укр. закінчень
        terms.append(f'"{stem}"*')
    return " OR ".join(terms)


def search(vault, q: str, embedder=None, k: int = 8, sources: list[str] | None = None):
    rng = parse_dates(q)
    flt, args = "", []
    if rng:
        flt += " AND d.ts >= ? AND d.ts < ?"; args += list(rng)
    if sources:
        flt += f" AND d.source IN ({','.join('?' * len(sources))})"; args += sources
    base = "FROM chunks c JOIN docs d ON d.id=c.doc_id"

    ranks: dict[int, float] = {}
    fq = fts_query(q)
    if fq:
        rows = vault.db.execute(
            f"SELECT fts.rowid FROM fts JOIN chunks c ON c.id=fts.rowid JOIN docs d ON d.id=c.doc_id "
            f"WHERE fts MATCH ? {flt} ORDER BY bm25(fts) LIMIT 30", [fq] + args).fetchall()
        for r, (cid,) in enumerate(rows):
            ranks[cid] = ranks.get(cid, 0) + 1 / (60 + r)
    if embedder:
        rows = vault.db.execute(f"SELECT c.id, c.emb {base} WHERE c.emb IS NOT NULL {flt}", args).fetchall()
        if rows:
            M = np.stack([np.frombuffer(e, dtype="float32") for _, e in rows])
            scores = M @ embedder.encode([q], query=True)[0]
            for r, i in enumerate(np.argsort(-scores)[:30]):
                ranks[rows[i][0]] = ranks.get(rows[i][0], 0) + 1 / (60 + r)

    hits, per_doc = [], {}
    for cid in sorted(ranks, key=ranks.get, reverse=True):
        text, doc_id, source, title, ts = vault.db.execute(
            f"SELECT c.text, d.id, d.source, d.title, d.ts {base} WHERE c.id=?", (cid,)).fetchone()
        if per_doc.get(doc_id, 0) >= 2:  # не більше 2 фрагментів з одного документа
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        hits.append(dict(n=len(hits) + 1, id=doc_id, source=source, title=title, ts=ts, text=text))
        if len(hits) == k:
            break
    return hits, rng
