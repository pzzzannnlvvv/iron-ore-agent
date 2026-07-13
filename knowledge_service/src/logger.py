"""日志:三段耗时 + 检索链路。用 stdlib logging,免装 loguru。"""
import logging
import sys


def get_logger(name: str = "knowledge_service") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()
