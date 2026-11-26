"""Конектори (лише читання): папка нотаток, IMAP, календар .ics."""
from __future__ import annotations
import datetime as dt, email, imaplib, os, re
from email import policy
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

import requests


class _Strip(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []
    def handle_data(self, d):
        self.parts.append(d)


def html2text(h: str) -> str:
    p = _Strip(); p.feed(h)
    return re.sub(r"[ \t]+", " ", " ".join(p.parts)).strip()


def sync_notes(vault, embedder, folder: str) -> int:
    n = 0
    for root, _, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.lower().endswith((".md", ".txt")):
                p = os.path.join(root, f)
                body = open(p, encoding="utf-8", errors="ignore").read()
                n += vault.upsert("note:" + os.path.relpath(p, folder), "note", os.path.splitext(f)[0],
                                  body, os.path.getmtime(p), embedder)
    vault.save()
    return n


def sync_imap(vault, embedder, host: str, user: str, password: str, folder: str = "INBOX",
              days: int = 365, limit: int = 300, progress=None) -> int:
    since = (dt.date.today() - dt.timedelta(days=days)).strftime("%d-%b-%Y")
    M = imaplib.IMAP4_SSL(host)
    M.login(user, password)
    M.select(f'"{folder}"', readonly=True)  # readonly: нічого не позначаємо прочитаним і не видаляємо
    ids = M.search(None, "SINCE", since)[1][0].split()[-limit:]
    n = 0
    for i, num in enumerate(ids):
        raw = M.fetch(num, "(BODY.PEEK[])")[1][0][1]
        m = email.message_from_bytes(raw, policy=policy.default)
        part = m.get_body(preferencelist=("plain", "html"))
        text = part.get_content() if part else ""
        if part and part.get_content_type() == "text/html":
            text = html2text(text)
        text = "\n".join(l for l in text.splitlines() if not l.startswith(">"))  # без цитат у відповідях
        try:
            ts = parsedate_to_datetime(m["Date"]).timestamp()
        except Exception:
            ts = dt.datetime.now().timestamp()
        body = f"Від: {m['From']}\nКому: {m['To']}\n\n{text}"
        n += vault.upsert("mail:" + (m["Message-ID"] or f"{host}:{num.decode()}"), "mail",
                          str(m["Subject"] or "(без теми)"), body, ts, embedder)
        if progress:
            progress((i + 1) / len(ids))
    M.logout()
    vault.save()
    return n


def sync_ics(vault, embedder, data: bytes) -> int:
    from icalendar import Calendar
    n = 0
    for ev in Calendar.from_ical(data).walk("VEVENT"):
        start = ev.decoded("DTSTART")
        if not isinstance(start, dt.datetime):
            start = dt.datetime.combine(start, dt.time())
        att = ", ".join(str(a).replace("mailto:", "") for a in (ev.get("ATTENDEE") if isinstance(ev.get("ATTENDEE"), list) else [ev.get("ATTENDEE")] if ev.get("ATTENDEE") else []))
        title = str(ev.get("SUMMARY", "(без назви)"))
        body = (f"Подія: {title}\nКоли: {start:%Y-%m-%d %H:%M}\nДе: {ev.get('LOCATION', '')}\n"
                f"Учасники: {att}\n\n{ev.get('DESCRIPTION', '')}")
        n += vault.upsert("cal:" + str(ev.get("UID", title + str(start))), "calendar", title, body,
                          start.timestamp(), embedder)
    vault.save()
    return n


def fetch_ics_url(url: str) -> bytes:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.content
