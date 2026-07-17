"""
复现2 · 报告生成客户端（阶段6）
调用 agent_service 的 /agent/api/deepresearch/stream，生成铁矿石库存分析报告。

=== 前提 ===
1. mcp_server 已启动：在 mcp_server/ 跑 `uv run python server.py`（监听 127.0.0.1:17000）
2. agent_service 已启动：在 agent_service/ 跑 `uv run uvicorn main:app --port 5000`
3. agent_service/.env.dev 已填你的 LLM 配置（LLM_BASE_URL/API_KEY/MODEL）

=== 用法（用 agent_service 的 venv 运行）===
    uv --directory ../agent_service run python scripts/generate_report.py "你的问题"
不传问题则用默认问题。
"""
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import httpx

AGENT_URL = "http://127.0.0.1:5000/agent/api/deepresearch/stream"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def handle_event(event_type, data, report_chunks, all_chunks, tool_events):
    """处理单个 SSE 事件。"""
    agent = data.get("agent", "") or data.get("langgraph_node", "")
    content = data.get("content", "")
    if event_type == "message_chunk":
        if content:
            all_chunks.append(content)
        if "reporter" in agent.lower():
            # reporter 节点的输出 = 最终报告，流式打印
            report_chunks.append(content)
            print(content, end="", flush=True)
    elif event_type == "tool_calls":
        tools = [t.get("name") for t in (data.get("tool_calls") or [])]
        if tools:
            print(f"\n[工具调用] agent={agent} tools={tools}", flush=True)
            tool_events.append({"agent": agent, "tools": tools})
    elif event_type == "tool_call_result":
        # 工具返回结果（researcher 的检索结果），可选打印前 80 字
        snippet = (content or "")[:80].replace("\n", " ")
        if snippet:
            print(f"  └ 工具返回: {snippet}...", flush=True)
    elif event_type == "error":
        print(f"\n[错误] {data.get('error')}", flush=True)


def generate_report(question: str, timeout: float = 600):
    """调用 agent 服务，流式接收并提取 reporter 的报告。"""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "agent_type": "default",
        "auto_accepted_plan": True,          # 自动接受计划，不中断等人审
        "enable_background_investigation": False,  # 跳过背景调查，加速
        "enable_clarification": False,       # 不多轮澄清
        "locale": "zh-CN",
        "max_step_num": 4,
    }
    report_chunks, all_chunks, tool_events = [], [], []
    print(f"▶ 提问: {question}")
    print(f"  (流式接收，约 1-3 分钟；研究过程如下)\n")
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", AGENT_URL, json=payload) as resp:
            resp.raise_for_status()
            event_type = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    handle_event(event_type, data, report_chunks, all_chunks, tool_events)
    report = "".join(report_chunks).strip()
    if not report:
        # fallback：reporter 没识别到时，拼所有 message_chunk
        report = "\n\n---\n\n".join(c for c in all_chunks if c)
    return report, tool_events


#基于铁矿石库存预测模型的结果，生成一份库存走势与分析报告，""包括模型表现（MDA、MAPE）和最近的预测值
def main():
    default_q = ("结合知识库里过往的铁矿石库存分析报告与项目预测数据，梳理本期库存走势与分析结论。")
    question = " ".join(sys.argv[1:]) or default_q
    report, tool_events = generate_report(question)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"report_{ts}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"\n\n✓ 报告已保存: {out_file}")
    print(f"  工具调用次数: {len(tool_events)}")


if __name__ == "__main__":
    main()
