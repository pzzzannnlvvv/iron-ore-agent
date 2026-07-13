"""入库管道:Load -> Split -> Embed(bge-m3) -> Store(Chroma),并重建 BM25 索引。"""
from __future__ import annotations

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import log
from src.pipeline.embedder import embed_batch
from src.pipeline.loader import load_corpus
from src.pipeline.splitter import split_text
from src.pipeline import store

_BATCH = 50


def ingest(corpus_dir=None) -> dict:
    store.reset()
    docs = load_corpus(corpus_dir)
    total = 0
    batch: list[dict] = []

    def flush():
        if batch:
            store.add_chunks(batch)
            batch.clear()

    for doc in docs:
        chunks = split_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        embs = embed_batch(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            batch.append({
                "id": f"{doc['doc_id']}_{i:04d}",
                "content": chunk,
                "embedding": emb,
                "metadata": {
                    "fileName": doc["file_name"],
                    "source": doc["source"],
                    "chunkIndex": i,
                    "sourceType": 1,
                },
            })
            if len(batch) >= _BATCH:
                flush()
        total += len(chunks)
        log.info("入库 %s: %d chunks", doc["file_name"], len(chunks))
    flush()

    # 重建 BM25(基于新入库的全量 chunk)
    from src.retrieval import bm25
    bm25.reset_index()

    log.info("入库完成: %d 文档, %d chunks", len(docs), total)
    return {"docs": len(docs), "chunks": total}
