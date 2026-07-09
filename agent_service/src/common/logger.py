import sys
from pathlib import Path

from loguru import logger

from src.config.loader import get_int_env, get_str_env


def configure_logging():
    """配置日志系统，支持stdout/stderr输出或文件输出（支持大小和时间轮换）"""
    # 移除默认的handler
    logger.remove()

    # 获取日志级别
    try:
        log_level = get_str_env("LOG_LEVEL", "").upper()
    except (AttributeError, ValueError):
        log_level = "INFO"

    if get_str_env("LOG_OUTPUT_TYPE", "") == "file":
        # 文件日志配置
        log_file_path = Path(get_str_env("LOG_FILE_PATH", "logs/xmschain_agent.log"))

        # 确保日志目录存在
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 配置日志轮换参数
        rotation_size = f"{get_int_env('LOG_FILE_MAX_SIZE', 50)}MB"
        backup_count = get_int_env("LOG_FILE_BACKUP_COUNT", 5)

        # 配置文件处理器 - 支持按大小轮换，时间轮换通过crontab或其他方式处理
        logger.add(
            str(log_file_path),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation_size,  # 按大小轮换
            retention=backup_count,  # 保留的日志文件数
            compression="zip",  # 压缩旧日志文件
            enqueue=True,  # 异步写入，提高性能
            backtrace=True,
            diagnose=True,
        )
    else:
        # 默认stdout输出（适配容器环境）
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    logger.info(
        f"Logging configured: output_type={get_str_env('LOG_OUTPUT_TYPE', '')}, level={log_level}"
    )
