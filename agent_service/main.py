import asyncio
import selectors
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import uvicorn
from src.patches import apply_all as apply_patches
from src.server import checkpointer_pool
from src.common.logger import configure_logging
from src.server.deepresearch import router as deepresearch_router
from src.server.scenarios import router as scenarios_router
from src.server.analysis import router as analysis_router
from src.server.suggested import router as suggested_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        asyncio.get_running_loop()
    except RuntimeError as e:
        logger.warning(f"Could not register asyncio exception handler: {e}")

    try:
        await checkpointer_pool.init()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    try:
        await checkpointer_pool.close()
        logger.info("Application shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logging()

# Apply runtime patches for third-party library bugs
apply_patches()

# Include API routes first
app.include_router(deepresearch_router)
app.include_router(scenarios_router)
app.include_router(analysis_router)
app.include_router(suggested_router)


async def main():
    """启动 Uvicorn 服务器"""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Windows 系统下使用 SelectorEventLoop
    if sys.platform == "win32":
        loop_factory = lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        asyncio.run(main(), loop_factory=loop_factory)
    else:
        asyncio.run(main())
