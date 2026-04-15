import json
from typing import Any, List, Optional


def to_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item.get("text")))
                elif item.get("content") is not None:
                    parts.append(str(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        if value.get("text") is not None:
            return str(value.get("text"))
        if value.get("content") is not None:
            return str(value.get("content"))
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def encode_text_tokens(text: str, tokenizer: Optional[Any]) -> List[int]:
    if not text:
        return []
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    return []


def decode_text_tokens(token_ids: List[int], tokenizer: Optional[Any]) -> str:
    if not token_ids:
        return ""
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    return ""


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Optional[Any],
    keep_tail: bool = False,
) -> str:
    if max_tokens is None or max_tokens <= 0 or not text:
        return ""
    ids = encode_text_tokens(text, tokenizer)
    if not ids:
        return text
    if len(ids) <= int(max_tokens):
        return text
    ids = ids[-int(max_tokens):] if keep_tail else ids[:int(max_tokens)]
    return decode_text_tokens(ids, tokenizer)


def truncate_prompt_to_max_tokens(prompt_text: str, max_prompt_tokens: int, tokenizer: Optional[Any]) -> str:
    if max_prompt_tokens is None or max_prompt_tokens <= 0 or not prompt_text:
        return prompt_text
    ids = encode_text_tokens(prompt_text, tokenizer)
    if not ids or len(ids) <= int(max_prompt_tokens):
        return prompt_text
    return decode_text_tokens(ids[: int(max_prompt_tokens)], tokenizer)


def count_tokens_text(text: str, tokenizer: Optional[Any]) -> int:
    return len(encode_text_tokens(text, tokenizer))
