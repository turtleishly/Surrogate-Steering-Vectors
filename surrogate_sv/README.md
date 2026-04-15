# surrogate_sv (Phase 1)

This package is the first extraction of notebook logic into Python modules.

## Current module groups

- `surrogate_sv.config`: constants + dataclass configs
- `surrogate_sv.paths`: Modal/local artifact path resolution
- `surrogate_sv.model`: HF reader cache, surrogate adapter, generation helper
- `surrogate_sv.prompt`: prompt parsing + monitor prompt builder
- `surrogate_sv.data`: Docent access + hidden-state extraction + artifact save helper

## Modal-friendly behavior

- Artifact root resolution checks:
  - `/root/datasets/artifacts/docent_activation_spaces`
  - `artifacts/docent_activation_spaces`
- Keep using `start_jupyter.py` volume mounts for `/root/datasets` and `/root/cache`.

## Minimal import example

```python
from surrogate_sv.config import COLLECTIONS, PromptPolicyConfig
from surrogate_sv.data.docent import get_docent_client
from surrogate_sv.data.extraction import extract_docent_transcript_means

client = get_docent_client()
policy = PromptPolicyConfig(max_prompt_tokens=4096, task_max_tokens=500)

tensor, meta = extract_docent_transcript_means(
    client=client,
    collection_map=COLLECTIONS,
    layer=12,
    prompt_policy=policy,
    reader_model_name="Qwen/Qwen3-8B",
    extraction_max_seq_len=4096,
    extraction_pooling="mean",
    max_transcripts_per_collection=10,
)
```

## Next migration step

Replace notebook helper definitions with imports from these modules, then verify parity on a fixed small slice.
