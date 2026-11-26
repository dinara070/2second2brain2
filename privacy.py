"""Оборотне маскування персональних даних перед відправкою в хмарну LLM."""
import re

PATTERNS = [
    ("IBAN", r"\bUA\d{2}[\dA-Z]{25}\b"),
    ("CARD", r"\b(?:\d[ -]?){13,19}\b"),
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("PHONE", r"\+?\d[\d\s().-]{8,}\d"),
]


class Redactor:
    def __init__(self):
        self.fwd: dict[str, str] = {}
        self.rev: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        for kind, pat in PATTERNS:
            def repl(m, kind=kind):
                v = m.group(0)
                if v not in self.fwd:
                    tok = f"[{kind}_{len(self.fwd) + 1}]"
                    self.fwd[v], self.rev[tok] = tok, v
                return self.fwd[v]
            text = re.sub(pat, repl, text)
        return text

    def restore(self, text: str) -> str:
        for tok, v in self.rev.items():
            text = text.replace(tok, v)
        return text
