import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain.agents import AgentState


PromptTree = Dict[str, Union[List[str], Dict[str, List[str]]]]


def load_prompts_tree() -> PromptTree:
    prompts_dir = Path("src/prompts")
    result: PromptTree = {}

    for category_dir in prompts_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if category_dir.name.startswith("__"):
            continue

        subdirs = [d for d in category_dir.iterdir() if d.is_dir()]

        if not subdirs:
            result[category_dir.name] = sorted(
                p.stem for p in category_dir.glob("*.md")
            )
        else:
            sub_map: Dict[str, List[str]] = {}
            for subdir in subdirs:
                sub_map[subdir.name] = sorted(p.stem for p in subdir.glob("*.md"))
            result[category_dir.name] = sub_map

    return result


# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def apply_prompt_template(
    prompt_name: str,
    state: AgentState,
    extra_variables: Optional[dict] = None,
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute
        extra_variables: Extra additional variables

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **state,
    }

    if extra_variables:
        state_vars.update(extra_variables)

    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)
        return [{"role": "system", "content": system_prompt}] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")


def render_system_prompt(
    prompt_name: str,
    extra_variables: Optional[dict] = None,
) -> str:
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **(extra_variables or {}),
    }
    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)
        return system_prompt
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
