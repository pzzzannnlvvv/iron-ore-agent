from typing import Any, Dict, Optional

import httpx
from loguru import logger
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.loader import get_str_env, get_int_env, get_bool_env
from src.config.agents import LLMType, AgentName, AGENT_TOKEN_LIMIT


# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}

# Allowed LLM configuration keys to prevent unexpected parameters from being passed
# to LLM constructors
ALLOWED_LLM_CONFIG_KEYS = {
    # Common LLM configuration keys
    "model",
    "api_key",
    "base_url",
    "api_base",
    "max_retries",
    "timeout",
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "streaming",
    "logprobs",
    "echo",
    "best_of",
    "logit_bias",
    "user",
    "seed",
    # SSL and HTTP client settings
    "verify_ssl",
    "http_client",
    "http_async_client",
    "extra_body",
    # Token limit for context compression (removed before passing to LLM)
    "token_limit",
    # Default headers
    "default_headers",
    "default_query",
}


class CustomChatOpenAI(ChatOpenAI):
    """自定义ChatOpenAI，支持通过extra_body传递thinking参数。"""

    thinking_enabled: Optional[bool] = None  # 可选：定义一个类属性来控制

    @property
    def _default_params(self) -> Dict[str, Any]:
        # 获取父类的默认参数
        default_params = super()._default_params

        # 准备要注入的额外请求体参数
        extra_body_dict = default_params.get("extra_body", {})
        # 将你的 thinking 配置合并进去
        # 这里可以根据 thinking_enabled 属性动态决定，目前硬编码为开启
        if get_bool_env("LLM_ENABLE_THINKING", False):
            extra_body_dict.update({"enable_thinking": True})
        else:
            extra_body_dict.update({"enable_thinking": False})
        extra_body_dict.update({"max_tokens": get_int_env("LLM_MAX_TOKENS", 4096)})
        # 将更新后的字典赋回给 extra_body 参数
        default_params["extra_body"] = extra_body_dict
        return default_params

class DoubaoChatOpenAI(ChatOpenAI):
    """自定义ChatOpenAI，支持通过extra_body传递thinking参数。"""

    thinking_enabled: Optional[bool] = None  # 可选：定义一个类属性来控制

    @property
    def _default_params(self) -> Dict[str, Any]:
        # 获取父类的默认参数
        default_params = super()._default_params

        # 准备要注入的额外请求体参数
        extra_body_dict = default_params.get("extra_body", {})
        # 将你的 thinking 配置合并进去
        # 这里可以根据 thinking_enabled 属性动态决定，目前硬编码为开启
        if get_bool_env("LLM_ENABLE_THINKING"):
            extra_body_dict.update({"thinking": {"type": "enabled"}})
        else:
            extra_body_dict.update({"thinking": {"type": "disabled"}})
        extra_body_dict.update({"max_tokens": get_int_env("LLM_MAX_TOKENS", 8192)})
        # 将更新后的字典赋回给 extra_body 参数
        default_params["extra_body"] = extra_body_dict
        return default_params


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    """
    return {
        "model": get_str_env("LLM_MODEL"),
        "base_url": get_str_env("LLM_BASE_URL"),
        "api_key": get_str_env("LLM_API_KEY"),
        "streaming": True if llm_type == "streaming" else False,
        "temperature": float(get_str_env("LLM_TEMPERATURE", "0.6")),
    }


def _create_llm_use_conf(llm_type: LLMType) -> BaseChatModel:
    """Create LLM instance using configuration."""
    env_conf = _get_env_llm_conf(llm_type)

    # Filter out unexpected parameters to prevent LangChain warnings (Issue #411)
    # This prevents configuration keys like SEARCH_ENGINE from being passed to LLM constructors
    allowed_keys_lower = {k.lower() for k in ALLOWED_LLM_CONFIG_KEYS}
    unexpected_keys = [
        key for key in env_conf.keys() if key.lower() not in allowed_keys_lower
    ]
    for key in unexpected_keys:
        logger.warning(
            f"Removed unexpected LLM configuration key '{key}'. "
            f"This key is not a valid LLM parameter and may have been placed in the wrong section of conf.yaml. "
            f"Valid LLM config keys include: model, api_key, base_url, max_retries, temperature, etc."
        )

    # Remove unnecessary parameters when initializing the client
    if "token_limit" in env_conf:
        env_conf.pop("token_limit")

    if not env_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Add max_retries to handle rate limit errors
    if "max_retries" not in env_conf:
        env_conf["max_retries"] = 3

    # Handle SSL verification settings
    verify_ssl = env_conf.pop("verify_ssl", True)

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        env_conf["http_client"] = http_client
        env_conf["http_async_client"] = http_async_client

    # 获取LLM供应商
    llm_vendor = get_str_env("LLM_VENDOR", "hiagent")
        
    if llm_vendor == "qwen":
        #按需增加其它参数
        env_conf["extra_body"] = {"enable_thinking": get_bool_env("LLM_ENABLE_THINKING", False)}
        env_conf["max_tokens"] = get_int_env("LLM_MAX_TOKENS", 8192)
        return ChatOpenAI(**env_conf)
    elif llm_vendor == "hiagent":
        return CustomChatOpenAI(**env_conf)
    elif llm_vendor == "deepseek":
        #按需增加其它参数
        if get_bool_env("LLM_ENABLE_THINKING"):
            env_conf["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            env_conf["extra_body"] = {"thinking": {"type": "disabled"}}
        env_conf["max_tokens"] = get_int_env("LLM_MAX_TOKENS", 8192)
        return ChatOpenAI(**env_conf)
    elif llm_vendor == "doubao":
        return DoubaoChatOpenAI(**env_conf)
    elif llm_vendor == "anthropic":
        # 火山方舟等 Anthropic 协议兼容端点（如 /api/coding），用 ChatAnthropic
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=env_conf["model"],
            anthropic_api_key=env_conf["api_key"],
            anthropic_api_url=env_conf["base_url"],
            temperature=env_conf["temperature"],
            max_tokens=get_int_env("LLM_MAX_TOKENS", 4096),
            max_retries=env_conf.get("max_retries", 3),
        )
    elif llm_vendor == "deepseekv4":
        # 为 deepseekv4 预设 extra_body，会被 Dsv4ChatOpenAI._default_params 识别
        env_conf["max_tokens"] = get_int_env("LLM_MAX_TOKENS", 4096)
        env_conf["extra_body"] = {
            "chat_template_kwargs": {
                "thinking": get_bool_env("LLM_ENABLE_THINKING", False)
            }
        }
        return ChatOpenAI(**env_conf)
    else:
        return CustomChatOpenAI(**env_conf)


def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    llm = _create_llm_use_conf(llm_type)
    _llm_cache[llm_type] = llm
    return llm


def _get_model_token_limit_defaults() -> dict[str, int]:
    """
    Get default token limits for common LLM models.
    These are conservative limits to prevent token overflow errors.
    Users can override by setting token_limit in their config.
    """
    return {
        # Default fallback for unknown models
        "default": 100000,
    }


def get_llm_token_limit_by_agent(agent_name: AgentName) -> int:
    """
    Get the maximum token limit for a given LLM agent.

    Priority order:
    1. Inferred from model name based on known model capabilities
    2. Safe default (100,000 tokens)

    Args:
        agent_name (str): The agent of LLM (e.g., 'reporter', 'planner').

    Returns:
        int: The maximum token limit for the specified LLM type (conservative estimate).
    """
    if AGENT_TOKEN_LIMIT.get(agent_name, 0) > 0:
        return AGENT_TOKEN_LIMIT[agent_name]
    else:
        return _get_model_token_limit_defaults()["default"]
