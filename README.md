# Surrogate Steering Vectors

This project builds and evaluates surrogate classifiers that detect malicious behavior from Docent trajectories.

The current system is notebook-orchestrated, with reusable Python modules in `surrogate_sv` handling data loading, split policy, prompt construction, activation extraction, and model training.

## Main Architecture

The architecture is split into three layers:

1. Orchestration layer
- `main_analysis_notebook.ipynb`
- Owns run configuration, stage execution order, caching, diagnostics, and ad hoc experiments.

2. Reusable pipeline layer
- `surrogate_sv/*`
- Encapsulates core operations:
  - data ingest from Docent
  - grouped split assignment with forced holdout
  - monitor prompt formatting
  - hidden-state extraction from HF reader
  - classifier training and evaluation

3. Runtime and storage layer
- `start_jupyter.py` defines Modal runtime, volumes, and GPU session setup.
- Artifacts are written to an output root resolved by `surrogate_sv.paths.resolve_output_root()`.

## End-to-End Data Flow

1. Index transcripts
- Source: Docent collections from `surrogate_sv.config.COLLECTIONS`
- Function: `surrogate_sv.experiments.build_transcript_index`
- Output: unified transcript index with metadata and messages.

2. Assign leak-safe grouped splits
- Function: `surrogate_sv.experiments.assign_grouped_splits`
- Group key: `agent_run_id` (fallback to transcript id if missing)
- Policy: selected collections are forced into holdout.

3. Build monitor prompts
- Function: `surrogate_sv.experiments.format_prompts_for_index`
- Prompt builder: `surrogate_sv.prompt.build_monitor_prompt_text`
- Applies prompt budget and tool-call selection policy.

4. Extract surrogate activations
- Function: `surrogate_sv.experiments.extract_activations_for_formatted`
- Backend extractor: `surrogate_sv.data.extraction.extract_layer_vector_for_text_hf`
- Supports multiple layers and pooling modes.

5. Train and evaluate binary classifiers
- Training: `surrogate_sv.classifiers.train_binary_models`
- Evaluation: `surrogate_sv.classifiers.evaluate_models`
- Models currently include LDA shrinkage and L2 logistic regression.

6. Report and inspect
- Notebook reporting cells generate:
  - metric tables
  - subtype diagnostics
  - ROC and PR curves
  - confusion matrices
  - false-classification spot checks with prompt text.

7. Single-trajectory inference path
- Notebook includes a cell for one-off inference:
  - optional monitor-format wrapping
  - activation extraction at selected layer
  - model prediction output (label, probability, margin)

## Stage Artifacts

Each run writes to:

- `artifacts/docent_activation_spaces/clean_pipeline_runs/<run_tag>/`

Typical outputs:

- `all_index_full.jsonl`
- `all_index_lite.csv`
- `split_assignments_full.jsonl`
- `split_assignments.csv`
- `formatted_prompts_full.jsonl`
- `formatted_prompts_lite.csv`
- `extraction_metadata.csv`
- `vectors_by_layer.npz`
- `metrics_all_layers.csv`
- `saved_models_index.csv`
- `saved_models/*.pkl`
- `pipeline_results.pkl`

## Caching and Run Modes

The notebook supports resume-first execution and overwrite workflows:

- `START_NEW_RUN`
  - create a new run tag.
- `RUN_TAG_OVERRIDE`
  - target a specific existing run.
- `OVERWRITE_EXISTING_RUN`
  - recompute and overwrite artifacts for selected run.

Each stage loads cached artifacts when present (unless overwrite is enabled), otherwise computes and persists outputs.

## Module Layout

- `surrogate_sv/config.py`
  - collection ids, prompt defaults, extraction defaults.
- `surrogate_sv/paths.py`
  - output root resolution for Modal/local contexts.
- `surrogate_sv/data/docent.py`
  - Docent client and paginated transcript iteration.
- `surrogate_sv/experiments/indexing.py`
  - transcript index assembly.
- `surrogate_sv/experiments/splits.py`
  - grouped split assignment and summaries.
- `surrogate_sv/experiments/formatting.py`
  - prompt formatting over indexed rows.
- `surrogate_sv/experiments/extraction.py`
  - activation extraction over formatted prompts.
- `surrogate_sv/prompt/*`
  - monitor prompt parsing, assembly, token-budget helpers.
- `surrogate_sv/classifiers/training.py`
  - model training and metric evaluation.
- `surrogate_sv/model/*`
  - HF adapter and generation utilities.

## Runtime Model

For remote GPU execution:

- start session with `modal run start_jupyter.py`
- mount code via Modal volume at `/workspace`
- mount datasets at `/root/datasets`
- mount model cache at `/root/cache`

This keeps heavy artifacts and model cache persistent while allowing notebook iteration.

## Quick Start

1. Launch Modal Jupyter with `start_jupyter.py`.
2. Open `main_analysis_notebook.ipynb`.
3. Run setup and staged cells in order.
4. Use reporting cells for diagnostics.
5. Use the single-trajectory inference cell for one-off classification checks.

## Design Notes

- Split policy is group-aware to reduce leakage.
- Forced-holdout subtype policy protects realistic evaluation.
- Prompt and extraction budgets are aligned to max context.
- Stage artifacts are designed for reproducibility, resuming, and post-hoc analysis.
