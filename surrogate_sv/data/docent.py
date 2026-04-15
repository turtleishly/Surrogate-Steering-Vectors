import json
import os
from typing import Any, Dict, Generator, Optional

try:
    from docent import Docent
except ImportError:  # pragma: no cover - environment dependent
    Docent = None


def get_docent_client() -> Any:
    if Docent is None:
        raise ImportError("Docent SDK unavailable. Install with: pip install docent-python")

    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DOCENT_API_KEY environment variable.")

    return Docent()


def build_docent_query(limit: int, offset: int, target_agent_run_id: Optional[str] = None) -> str:
    where_clause = ""
    if target_agent_run_id:
        where_clause = f"WHERE ar.id = '{target_agent_run_id}'"
    return f"""
SELECT
  t.id AS transcript_id,
  t.name AS transcript_name,
  t.messages,
  t.metadata_json AS transcript_metadata,
  ar.id AS agent_run_id,
  ar.name AS agent_run_name,
  ar.metadata_json AS agent_run_metadata
FROM transcripts t
JOIN agent_runs ar ON ar.id = t.agent_run_id
{where_clause}
ORDER BY t.id
LIMIT {limit} OFFSET {offset}
"""


def normalize_messages_field(row: Dict[str, Any]) -> Dict[str, Any]:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            row["messages"] = json.loads(messages)
        except json.JSONDecodeError:
            row["messages"] = []
    return row


def iter_docent_transcripts(
    client: Any,
    collection_id: str,
    page_size: int,
    target_agent_run_id: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    offset = 0
    while True:
        query = build_docent_query(limit=page_size, offset=offset, target_agent_run_id=target_agent_run_id)
        result = client.execute_dql(collection_id, query)
        rows = client.dql_result_to_dicts(result)
        if not rows:
            break
        for row in rows:
            yield normalize_messages_field(row)
        offset += page_size
