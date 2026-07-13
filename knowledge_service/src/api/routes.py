"""FastAPI 路由:/knowledge-base/report/search、/knowledge-base/index、/health。"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import IndexResponse, KnowledgeSearchRequest, SearchResponse
from src.logger import log
from src.pipeline import store
from src.pipeline.ingest import ingest
from src.retrieval.search import search as do_search

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "chunks": store.count()}


@router.post("/knowledge-base/report/search")
def report_search(req: KnowledgeSearchRequest):
    hits, timings = do_search(
        content=req.content,
        topK=req.topK,
        reRank=req.reRank,
        mode=req.mode,
        publishStartDate=req.publishStartDate,
        publishEndDate=req.publishEndDate,
        sourceType=req.sourceType,
    )
    return SearchResponse(data=hits, timings=timings).model_dump()


@router.post("/knowledge-base/index")
def index():
    """触发重新入库(读 data/corpus -> 切片 -> 编码 -> 写 Chroma)。"""
    log.info("收到 /knowledge-base/index 请求,开始入库...")
    result = ingest()
    return IndexResponse(data=result).model_dump()
