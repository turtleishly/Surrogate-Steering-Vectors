from typing import Any, Dict

import pandas as pd

from ..data.docent import iter_docent_transcripts


def _count_tool_calls(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for key in ["tool_calls", "calls", "tools"]:
            val = msg.get(key)
            if isinstance(val, list):
                total += len(val)
    return total


def build_transcript_index(
    client: Any,
    collection_map: Dict[str, str],
    page_size: int = 200,
) -> pd.DataFrame:
    """Load all available Docent transcripts and return a flat index table."""
    rows = []
    for collection_name, collection_id in collection_map.items():
        for row in iter_docent_transcripts(client, collection_id=collection_id, page_size=page_size):
            messages = row.get("messages")
            tool_call_count = _count_tool_calls(messages)
            rows.append(
                {
                    "collection_name": collection_name,
                    "collection_id": collection_id,
                    "transcript_id": row.get("transcript_id"),
                    "transcript_name": row.get("transcript_name"),
                    "agent_run_id": row.get("agent_run_id"),
                    "agent_run_name": row.get("agent_run_name"),
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "tool_call_count": int(tool_call_count),
                    "has_messages": bool(isinstance(messages, list) and len(messages) > 0),
                    "has_tool_calls": bool(tool_call_count > 0),
                    "messages": messages,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["collection_name", "transcript_id"]).reset_index(drop=True)
    return df
