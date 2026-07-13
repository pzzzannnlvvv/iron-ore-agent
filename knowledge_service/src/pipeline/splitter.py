"""
Split:递归字符切片,中文友好。
按 段落 > 换行 > 句号/问号/叹号 > 分号 > 逗号 > 空格 逐级切,直到片段 <= chunk_size,
再贪心合并成 chunk,带 overlap。对应方案 splitter.py(参考原版 FileContentSplitter)。
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", "。", "!", "?", ";", "!", "?", ";", ",", "，", " ", ""]


def _split_recursively(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep, rest = separators[0], separators[1:]
    out: list[str] = []
    for piece in text.split(sep):
        if len(piece) > chunk_size:
            out.extend(_split_recursively(piece, rest, chunk_size))
        else:
            out.append(piece)
    return out


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    separators = separators or SEPARATORS
    pieces = _split_recursively(text, separators, chunk_size)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + len(p) <= chunk_size:
            cur += p
        else:
            chunks.append(cur)
            tail = cur[-chunk_overlap:] if chunk_overlap > 0 else ""
            cur = (tail + p) if tail else p
    if cur.strip():
        chunks.append(cur)
    return chunks
