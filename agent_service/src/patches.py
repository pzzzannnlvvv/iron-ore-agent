"""
Monkey patches for third-party library bugs.

Applied at application startup (main.py) to fix issues in dependencies
that we can't / shouldn't modify directly in site-packages.
"""

from loguru import logger
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def _patch_summarization_middleware():
    """
    Patch SummarizationMiddleware.before_model / abefore_model to ensure
    preserved_messages always contains at least one HumanMessage (the latest
    user question or instruction).

    Original bug chain:
    1. _find_safe_cutoff_point skips ToolMessages, may advance to len(messages)
       → preserved_messages == [] (all context lost)
    2. Even when preserved is non-empty, it's often just raw ToolMessage entries
       → model sees: [System, Human(summary), Tool(raw_data)] — no user question

    Fix: After the original summary logic runs, scan preserved_messages.
    If no HumanMessage is present, walk backwards through the full message list
    to locate the latest HumanMessage, then rebuild the result with it included.
    """
    from langchain.agents.middleware.summarization import SummarizationMiddleware

    original_before = SummarizationMiddleware.before_model
    original_abefore = SummarizationMiddleware.abefore_model

    def _ensure_human_in_preserved(
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> int:
        """Walk backwards from cutoff_index to include the latest HumanMessage."""
        preserved = messages[cutoff_index:]
        if any(isinstance(m, HumanMessage) for m in preserved):
            return cutoff_index
        for i in range(cutoff_index - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                logger.info(
                    f"[patches] Extended preserved messages: cutoff adjusted "
                    f"from {cutoff_index} to {i} to include user message"
                )
                return i
        return 0

    def _rebuild_with_preserved(
        summary_msg: HumanMessage,
        preserved: list[AnyMessage],
        state_messages: list[AnyMessage],
    ) -> dict:
        """Rebuild the middleware result ensuring at least one HumanMessage in preserved."""
        if any(isinstance(m, HumanMessage) for m in preserved):
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    summary_msg,
                    *preserved,
                ]
            }

        # Preserved has no HumanMessage — find the latest one in state
        preserved_ids = {m.id for m in preserved if m.id}
        last_human_idx = -1
        for i, sm in enumerate(state_messages):
            if isinstance(sm, HumanMessage) and (not sm.id or sm.id not in preserved_ids):
                last_human_idx = i

        if last_human_idx >= 0:
            new_cutoff = _ensure_human_in_preserved(state_messages, last_human_idx)
            _, new_preserved = state_messages[:new_cutoff], state_messages[new_cutoff:]
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    summary_msg,
                    *new_preserved,
                ]
            }

        # Extreme fallback — keep last 5 messages no matter what
        fallback = state_messages[-5:] if len(state_messages) >= 5 else state_messages
        logger.warning(
            f"[patches] No HumanMessage found in state, keeping fallback {len(fallback)} messages"
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *fallback,
            ]
        }

    def patched_before_model(self, state, runtime):
        result = original_before(self, state, runtime)
        if result is None:
            return None

        new_messages = result.get("messages", [])
        if not new_messages:
            return result

        non_remove = [m for m in new_messages if not isinstance(m, RemoveMessage)]
        summary_msg = None
        preserved = []
        for m in non_remove:
            if isinstance(m, HumanMessage) and m.content.startswith(
                "Here is a summary of the conversation to date"
            ):
                summary_msg = m
            else:
                preserved.append(m)

        if summary_msg:
            return _rebuild_with_preserved(summary_msg, preserved, state.get("messages", []))
        return result

    async def patched_abefore_model(self, state, runtime):
        result = await original_abefore(self, state, runtime)
        if result is None:
            return None

        new_messages = result.get("messages", [])
        if not new_messages:
            return result

        non_remove = [m for m in new_messages if not isinstance(m, RemoveMessage)]
        summary_msg = None
        preserved = []
        for m in non_remove:
            if isinstance(m, HumanMessage) and m.content.startswith(
                "Here is a summary of the conversation to date"
            ):
                summary_msg = m
            else:
                preserved.append(m)

        if summary_msg:
            return _rebuild_with_preserved(summary_msg, preserved, state.get("messages", []))
        return result

    SummarizationMiddleware.before_model = patched_before_model
    SummarizationMiddleware.abefore_model = patched_abefore_model
    logger.info(
        "[patches] SummarizationMiddleware patched: "
        "preserved_messages always includes at least one HumanMessage"
    )


def apply_all():
    """Call at startup to apply all monkey patches."""
    _patch_summarization_middleware()
