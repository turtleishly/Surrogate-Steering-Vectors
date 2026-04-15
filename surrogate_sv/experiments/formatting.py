from typing import Any, Optional

import pandas as pd
from tqdm.auto import tqdm

from ..config import DEFAULT_READER_MODEL_NAME, PromptPolicyConfig
from ..model.adapter import get_hf_reader
from ..prompt.builder import build_monitor_prompt_text
from ..prompt.text_utils import count_tokens_text


def format_prompts_for_index(
    index_df: pd.DataFrame,
    prompt_policy: PromptPolicyConfig,
    reader_model_name: str = DEFAULT_READER_MODEL_NAME,
    tokenizer: Optional[Any] = None,
    drop_no_tool_calls: bool = True,
) -> pd.DataFrame:
    """Build monitor prompts for indexed transcripts and append prompt metadata."""
    if index_df.empty:
        return index_df.copy()

    tok = tokenizer
    if tok is None:
        _, tok = get_hf_reader(model_name=reader_model_name)

    rows = []
    records = index_df.to_dict(orient="records")
    iterator = tqdm(records, total=len(records), desc="Formatting prompts")
    for rec in iterator:
        messages = rec.get("messages")

        prompt_text, kept_tool_calls, has_user_task = build_monitor_prompt_text(
            messages=messages,
            policy=prompt_policy,
            tokenizer=tok,
        )

        if drop_no_tool_calls and kept_tool_calls == 0:
            continue

        row = dict(rec)
        row.update(
            {
                "prompt_text": prompt_text,
                "prompt_char_len": len(prompt_text) if prompt_text else 0,
                "prompt_token_len": int(count_tokens_text(prompt_text, tok)) if prompt_text else 0,
                "kept_tool_calls": int(kept_tool_calls),
                "has_user_task": bool(has_user_task),
                "has_prompt": bool(prompt_text),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
