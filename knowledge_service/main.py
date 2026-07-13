"""
knowledge_service FastAPI 入口(:8092)。

启动:
    .venv/Scripts/python.exe -m uvicorn main:app --port 8092 --host 127.0.0.1
或:
    .venv/Scripts/python.exe main.py
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import SERVICE_PORT
from src.logger import log
from src.pipeline import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("knowledge_service 启动 port=%d 已入库 chunks=%d", SERVICE_PORT, store.count())
    yield
    log.info("knowledge_service 关闭")


app = FastAPI(title="knowledge_service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=SERVICE_PORT)
