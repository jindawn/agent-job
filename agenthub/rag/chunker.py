"""Heading-aware markdown chunker with overlap.

Each chunk keeps its heading path ("文档 > 小节") so retrieval results are citable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return f"{self.doc} > {self.heading}" if self.heading else self.doc


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown(text: str, doc: str, max_chars: int = 500, overlap: int = 60) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path[1:]) if len(path) > 1 else (path[0] if path else ""), [body]))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, bodies in sections:
        for body in bodies:
            for piece in _window(body, max_chars, overlap):
                cid = hashlib.md5(f"{doc}|{heading}|{piece}".encode()).hexdigest()[:10]
                chunks.append(Chunk(id=cid, doc=doc, heading=heading, text=piece))
    return chunks


def _window(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    paras = [p.strip() for p in re.split(r"\n\s*\n|(?<=[。！？!?])", body) if p.strip()]
    out: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur)
            cur = cur[-overlap:] + p if overlap else p
        else:
            cur += p
    if cur:
        out.append(cur)
    return out


def load_dir(directory: Path, **kw) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(Path(directory).glob("**/*.md")):
        chunks.extend(split_markdown(p.read_text(encoding="utf-8"), doc=p.name, **kw))
    return chunks
