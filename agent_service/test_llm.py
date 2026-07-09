"""测试 LLM 配置是否调通（阶段1验证）。
用法：uv run python test_llm.py
应输出 LLM_OK: '...' 表示火山方舟 GLM 调通。
"""
import asyncio

import src.config  # 触发 load_dotenv(.env.dev)
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP


async def main():
    llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
    res = await llm.ainvoke("说一句你好，只回复这一句")
    print("LLM_OK:", repr(res.content[:200]))


if __name__ == "__main__":
    asyncio.run(main())
