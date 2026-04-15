from typing import Dict, Iterable, cast

import pandas as pd
import torch as t
from tqdm.auto import tqdm

from ..data.extraction import PoolingMode, extract_layer_vector_for_text_hf


def extract_activations_for_formatted(
    formatted_df: pd.DataFrame,
    layers: Iterable[int],
    pooling: str,
    reader_model_name: str,
    max_seq_len: int,
) -> Dict[str, object]:
    """Extract activations for formatted prompts for one or many layers."""
    if formatted_df.empty:
        return {
            "metadata": formatted_df.copy(),
            "vectors_by_layer": {int(layer): t.empty(0) for layer in layers},
        }

    layer_list = [int(x) for x in layers]
    vectors_by_layer = {layer: [] for layer in layer_list}
    kept_rows = []

    pooling_mode = cast(PoolingMode, pooling)

    records = formatted_df.to_dict(orient="records")
    iterator = tqdm(records, total=len(records), desc="Extracting activations")
    for rec in iterator:
        prompt_text = rec.get("prompt_text", "")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue

        try:
            row_vectors = {}
            token_count = None
            for layer in layer_list:
                vec, token_count_local = extract_layer_vector_for_text_hf(
                    text=prompt_text,
                    layer=layer,
                    max_seq_len=max_seq_len,
                    pooling=pooling_mode,
                    model_name=reader_model_name,
                )
                row_vectors[layer] = vec
                token_count = token_count_local

            for layer in layer_list:
                vectors_by_layer[layer].append(row_vectors[layer])

            row = dict(rec)
            row["extraction_token_len"] = int(token_count) if token_count is not None else 0
            kept_rows.append(row)
        except Exception as exc:
            print(f"skip transcript_id={rec.get('transcript_id')} due to extraction error: {exc}")
            continue

    out_vectors = {}
    for layer, vectors in vectors_by_layer.items():
        out_vectors[layer] = t.stack(vectors, dim=0).to(dtype=t.float32) if vectors else t.empty(0)

    return {
        "metadata": pd.DataFrame(kept_rows),
        "vectors_by_layer": out_vectors,
    }
