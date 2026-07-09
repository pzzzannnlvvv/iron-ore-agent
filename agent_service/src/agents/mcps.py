from datetime import timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config.loader import get_int_env


async def merge_mcp_tools(
    agent_name: str,
    mcp_settings: dict,
    default_tools: list,
) -> tuple[bool, list]:
    mcp_servers = {}
    enabled_tools = {}

    for server_name, server_config in mcp_settings["servers"].items():
        for i in server_config["enabled_tools"]:
            if agent_name == i["node"]:
                tmp_config = {
                    k: v
                    for k, v in server_config.items()
                    if k
                    in (
                        "transport",
                        "command",
                        "args",
                        "url",
                        "env",
                        "headers",
                        "timeout",
                        "sse_read_timeout",
                    )
                }
                tmp_config["timeout"] = timedelta(
                    seconds=tmp_config["timeout"]
                    if tmp_config.get("timeout", None)
                    else get_int_env("MCP_DEFAULT_TIMEOUT")
                )
                tmp_config["sse_read_timeout"] = timedelta(
                    seconds=tmp_config["sse_read_timeout"]
                    if tmp_config.get("sse_read_timeout", None)
                    else get_int_env("MCP_DEFAULT_SSE_READ_TIMEOUT")
                )
                mcp_servers[server_name] = tmp_config
                for tool_name in i["tools"]:
                    enabled_tools[tool_name] = server_name

    # Create and execute agent with MCP tools if available
    if mcp_servers:
        client = MultiServerMCPClient(mcp_servers)
        loaded_tools = default_tools[:]
        all_tools = await client.get_tools()
        for tool in all_tools:
            if tool.name in enabled_tools:
                tool.description = (
                    f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                )
                loaded_tools.append(tool)

        return True, loaded_tools
    else:
        return False, default_tools
