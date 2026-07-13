"""
Load:读 data/corpus 下的 .md/.txt/.html,返回 [{doc_id, file_name, source, content}]。

.html 用 stdlib html.parser 抽纯文本(跳过 script/style),不引入 bs4 依赖。
对应 knowledge_be接入方案 第五节 loader.py。
"""
from __future__ import annotations

import html as html_lib
import re
from html.parser import HTMLParser
from pathlib import Path

from src.config import CORPUS_DIR


class _TextExtractor(HTMLParser):
    """从 HTML 抽纯文本,跳过 script/style。"""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = html_lib.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html_str: str) -> str:
    return _TextExtractor().feed_and_get(html_str) if hasattr(_TextExtractor, "feed_and_get") else _extract(html_str)


def _extract(html_str: str) -> str:
    p = _TextExtractor()
    p.feed(html_str)
    return p.get_text()


def load_corpus(corpus_dir: Path | None = None) -> list[dict]:
    """遍历 corpus_dir 下所有 .md/.txt/.html,返回文档列表。"""
    corpus_dir = corpus_dir or CORPUS_DIR
    docs: list[dict] = []
    if not corpus_dir.exists():
        return docs
    for i, path in enumerate(sorted(corpus_dir.glob("**/*"))):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if ext == ".html":
            content = _extract(raw)
        elif ext in (".md", ".txt"):
            content = raw
        else:
            continue
        if not content.strip():
            continue
        docs.append({
            "doc_id": f"doc_{i:03d}",
            "file_name": path.name,
            "source": str(path.relative_to(corpus_dir)),
            "content": content,
        })
    return docs
