from typing import Any, Dict, List, Optional, Tuple

from .text_utils import to_text_value


def extract_first_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() != "user":
            continue
        text = to_text_value(msg.get("content")).strip()
        if text:
            return text
    return ""


def parse_tool_call_dict(raw_call: Dict[str, Any]) -> Dict[str, str]:
    tool_name = (
        raw_call.get("tool")
        or raw_call.get("function")
        or raw_call.get("name")
        or raw_call.get("tool_name")
        or "tool"
    )
    raw_input = raw_call.get("input")
    if raw_input is None:
        raw_input = raw_call.get("arguments")
    if raw_input is None:
        raw_input = raw_call.get("args")
    return {
        "tool_name": str(tool_name),
        "tool_input": to_text_value(raw_input),
    }


def extract_tool_calls_from_messages(
    messages: Any,
    *,
    first_n: Optional[int],
) -> Tuple[str, int, int]:
    """Return assistant tool-call inputs with optional cap."""
    if not isinstance(messages, list):
        return "", 0, 0

    parsed_inputs: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for key in ["tool_calls", "calls", "tools"]:
            call_list = msg.get(key)
            if not isinstance(call_list, list):
                continue
            for raw_call in call_list:
                if not isinstance(raw_call, dict):
                    continue
                parsed = parse_tool_call_dict(raw_call)
                in_text = (parsed.get("tool_input") or "").strip()
                if not in_text:
                    continue
                tool_name = parsed.get("tool_name", "tool")
                parsed_inputs.append(f"tool={tool_name}\ninput\n```text\n{in_text}\n```")

    total_calls = len(parsed_inputs)
    if total_calls == 0:
        return "", 0, 0

    if first_n is None:
        kept = parsed_inputs
    else:
        keep_n = max(0, int(first_n))
        kept = parsed_inputs[:keep_n]

    return "\n\n".join(kept).strip(), len(kept), total_calls
