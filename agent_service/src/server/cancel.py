from fastapi import APIRouter

from src.common.aredis import async_redis_manager
from .schemas import ChatCancel
from .utils import CANCEL_KEY


router = APIRouter(prefix="/agent/api", tags=["cancel"])


@router.post("/cancel")
async def chat_cancel(request: ChatCancel):
    await async_redis_manager.client.set(
        f"{CANCEL_KEY}:{request.thread_id}", 1, ex=60 * 60, nx=True
    )
    return {"code": 0, "msg": "success", "info": None}
