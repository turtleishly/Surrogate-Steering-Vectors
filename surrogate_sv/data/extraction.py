import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch as t

from ..config import PromptPolicyConfig
from ..model.adapter import get_hf_reader
from ..paths import resolve_output_root
from ..prompt.builder import build_monitor_prompt_text
from .docent import iter_docent_transcripts

PoolingMode = Literal["last_token", "mean"]


def extract_layer_vector_for_text_hf(
    text: str,
    layer: int,
    max_seq_len: int,
    pooling: PoolingMode,
    model_name: str,
) -> Tuple[t.Tensor, int]:
    if not text.strip():
        raise ValueError("Empty text provided for activation extraction.")

    model, tokenizer = get_hf_reader(model_name=model_name)
    device_local = next(model.parameters()).device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_seq_len),
        padding=False,
    )
    input_ids = enc["input_ids"].to(device_local)
    attention_mask = enc["attention_mask"].to(device_local)

    with t.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = out.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states")

    h = hidden_states[layer]

    if pooling == "last_token":
        last_idx = attention_mask.sum(dim=1) - 1
        batch_ar = t.arange(h.size(0), device=h.device)
        vec = h[batch_ar, last_idx, :]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (h * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        vec = summed / lengths
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    token_count = int(attention_mask.sum().item())
    return vec[0].detach().to("cpu", dtype=t.float32), token_count


def extract_docent_transcript_means(
    client: Any,
    collection_map: Dict[str, str],
    layer: int,
    prompt_policy: PromptPolicyConfig,
    reader_model_name: str,
    extraction_max_seq_len: int,
    extraction_pooling: PoolingMode = "last_token",
    page_size: int = 200,
    max_transcripts_per_collection: Optional[int] = None,
    verbose_every: int = 25,
) -> Tuple[t.Tensor, pd.DataFrame]:
    _, prompt_tokenizer = get_hf_reader(model_name=reader_model_name)

    means: List[t.Tensor] = []
    rows_meta: List[Dict[str, Any]] = []

    for collection_name, collection_id in collection_map.items():
        kept = 0
        skipped_no_tools = 0
        seen = 0

        print(f"\n=== Collection: {collection_name} ({collection_id}) ===")

        for transcript in iter_docent_transcripts(client, collection_id=collection_id, page_size=page_size):
            if max_transcripts_per_collection is not None and seen >= max_transcripts_per_collection:
                break

            seen += 1
            messages = transcript.get("messages")

            prompt_text, tool_call_count, has_user_task = build_monitor_prompt_text(
                messages=messages,
                policy=prompt_policy,
                tokenizer=prompt_tokenizer,
            )
            if not prompt_text or tool_call_count == 0:
                skipped_no_tools += 1
                continue

            try:
                mean_vec, token_count = extract_layer_vector_for_text_hf(
                    text=prompt_text,
                    layer=layer,
                    max_seq_len=extraction_max_seq_len,
                    pooling=extraction_pooling,
                    model_name=reader_model_name,
                )
            except Exception as exc:
                print(f"skip transcript due to extraction error: {exc}")
                continue

            means.append(mean_vec)
            kept += 1

            rows_meta.append(
                {
                    "collection_name": collection_name,
                    "collection_id": collection_id,
                    "transcript_id": transcript.get("transcript_id"),
                    "transcript_name": transcript.get("transcript_name"),
                    "agent_run_id": transcript.get("agent_run_id"),
                    "agent_run_name": transcript.get("agent_run_name"),
                    "tool_call_count": tool_call_count,
                    "has_user_task": bool(has_user_task),
                    "prompt_char_len": len(prompt_text),
                    "prompt_token_len": token_count,
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "extraction_pooling": extraction_pooling,
                    "extraction_max_seq_len": int(extraction_max_seq_len),
                    "prompt_budget_tokens": int(prompt_policy.max_prompt_tokens),
                    "transcript_metadata": json.dumps(transcript.get("transcript_metadata", {}), ensure_ascii=False),
                    "agent_run_metadata": json.dumps(transcript.get("agent_run_metadata", {}), ensure_ascii=False),
                }
            )

            if verbose_every > 0 and kept % verbose_every == 0:
                print(
                    f"kept={kept} seen={seen} skipped_no_tools={skipped_no_tools} "
                    f"(collection={collection_name})"
                )

        print(f"Finished {collection_name}: kept={kept}, seen={seen}, skipped_no_tools={skipped_no_tools}")

    if not means:
        raise RuntimeError("No transcript vectors were produced. Check prompt composition and extraction backend.")

    transcript_mean_tensor = t.stack(means, dim=0)
    metadata_df = pd.DataFrame(rows_meta)

    print(f"\nTotal transcripts kept: {len(metadata_df)}")
    print(f"transcript_mean_tensor shape: {tuple(transcript_mean_tensor.shape)}")

    return transcript_mean_tensor, metadata_df


def save_docent_activation_artifacts(
    transcript_mean_tensor: t.Tensor,
    metadata_df: pd.DataFrame,
    run_tag: str,
    output_root: Optional[Path] = None,
) -> Dict[str, Path]:
    root = output_root if output_root is not None else resolve_output_root()
    root.mkdir(parents=True, exist_ok=True)

    tensor_path = root / f"{run_tag}_transcript_mean.pt"
    meta_path = root / f"{run_tag}_metadata.csv"
    manifest_path = root / f"{run_tag}_manifest.json"

    t.save(transcript_mean_tensor, tensor_path)
    metadata_df.to_csv(meta_path, index=False)

    manifest = {
        "run_tag": run_tag,
        "num_rows": int(metadata_df.shape[0]),
        "d_model": int(transcript_mean_tensor.shape[1]),
        "collections": sorted(metadata_df["collection_name"].dropna().unique().tolist()) if "collection_name" in metadata_df else [],
        "tensor_file": str(tensor_path),
        "metadata_file": str(meta_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "tensor_path": tensor_path,
        "metadata_path": meta_path,
        "manifest_path": manifest_path,
    }
