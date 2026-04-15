from typing import Any, Optional, Tuple

from ..config import PromptPolicyConfig
from .parsing import extract_first_user_message, extract_tool_calls_from_messages
from .text_utils import count_tokens_text, truncate_prompt_to_max_tokens, truncate_text_to_tokens


def build_monitor_prompt_text(
    messages: Any,
    policy: PromptPolicyConfig,
    tokenizer: Optional[Any] = None,
) -> Tuple[str, int, bool]:
    """Build monitor prompt preserving notebook semantics and budgets."""
    user_text = ""
    if policy.include_first_user_message:
        raw_user = extract_first_user_message(messages)
        user_text = truncate_text_to_tokens(
            raw_user,
            int(policy.task_max_tokens),
            tokenizer,
            keep_tail=False,
        )

    raw_tool_text, kept_tool_calls, total_tool_calls = extract_tool_calls_from_messages(
        messages,
        first_n=policy.first_n_tool_calls,
    )

    sys_text = (policy.system_prompt or "").strip() if policy.include_system_prompt else ""
    tool_prefix = f"Trajectory tool activity (first {kept_tool_calls}/{total_tool_calls} calls):\n"

    def render_chat(user_tool_text: str) -> str:
        chat_messages = []
        if sys_text:
            chat_messages.append({"role": "system", "content": sys_text})

        user_parts = []
        if user_text:
            user_parts.append("Task summary:\n" + user_text)
        if user_tool_text:
            user_parts.append(tool_prefix + user_tool_text)
        if user_parts:
            chat_messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        if (
            policy.use_chat_template
            and tokenizer is not None
            and chat_messages
            and hasattr(tokenizer, "apply_chat_template")
        ):
            return str(
                tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=policy.add_generation_prompt,
                )
            ).strip()

        parts = []
        if sys_text:
            parts.append("system\n```text\n" + sys_text + "\n```")
        if user_text:
            parts.append("user_task_summary\n```text\n" + user_text + "\n```")
        if user_tool_text:
            parts.append(f"tool_activity_first_{kept_tool_calls}_of_{total_tool_calls}\n" + user_tool_text)
        return "\n\n".join(parts).strip()

    scaffold_prompt = render_chat("")
    scaffold_tokens = count_tokens_text(scaffold_prompt, tokenizer)
    remaining_for_tools = max(0, int(policy.max_prompt_tokens) - scaffold_tokens)

    tool_text = (
        truncate_text_to_tokens(
            raw_tool_text,
            max_tokens=remaining_for_tools,
            tokenizer=tokenizer,
            keep_tail=False,
        )
        if raw_tool_text
        else ""
    )

    prompt_text = render_chat(tool_text)
    prompt_text = truncate_prompt_to_max_tokens(
        prompt_text=prompt_text,
        max_prompt_tokens=policy.max_prompt_tokens,
        tokenizer=tokenizer,
    )
    return prompt_text, kept_tool_calls, bool(user_text)
