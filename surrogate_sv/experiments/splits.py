from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.1
    holdout_frac: float = 0.2
    random_state: int = 42
    group_col: str = "agent_run_id"
    transcript_col: str = "transcript_id"
    collection_col: str = "collection_name"
    forced_holdout_collection: Optional[str] = "execute_malware"


def _resolve_groups(df: pd.DataFrame, group_col: str, transcript_col: str) -> pd.Series:
    groups = df[group_col].astype(str).copy()
    missing = df[group_col].isna() | (groups.str.strip() == "") | (groups == "None")
    groups.loc[missing] = df.loc[missing, transcript_col].astype(str)
    return groups


def assign_grouped_splits(index_df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    if index_df.empty:
        return index_df.copy()

    total = config.train_frac + config.val_frac + config.holdout_frac
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    work = index_df.copy().reset_index(drop=True)
    work["split"] = "unassigned"

    forced_mask = pd.Series(False, index=work.index)
    if config.forced_holdout_collection is not None:
        forced_mask = work[config.collection_col].astype(str) == str(config.forced_holdout_collection)
        work.loc[forced_mask, "split"] = "holdout"

    candidate = work.loc[~forced_mask].copy()
    if candidate.empty:
        return work

    groups = _resolve_groups(candidate, config.group_col, config.transcript_col)

    gss_1 = GroupShuffleSplit(n_splits=1, test_size=(config.val_frac + config.holdout_frac), random_state=config.random_state)
    train_rel, temp_rel = next(gss_1.split(candidate, groups=groups))

    train_idx = candidate.index[train_rel]
    temp_df = candidate.iloc[temp_rel].copy()

    # Split remaining chunk into val/holdout using target ratio val : holdout.
    temp_groups = _resolve_groups(temp_df, config.group_col, config.transcript_col)
    val_share_of_temp = config.val_frac / (config.val_frac + config.holdout_frac)

    gss_2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_share_of_temp), random_state=config.random_state)
    val_rel, hold_rel = next(gss_2.split(temp_df, groups=temp_groups))

    val_idx = temp_df.index[val_rel]
    hold_idx = temp_df.index[hold_rel]

    work.loc[train_idx, "split"] = "train"
    work.loc[val_idx, "split"] = "val"
    work.loc[hold_idx, "split"] = "holdout"

    return work


def summarize_split_counts(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame(columns=["split", "collection_name", "count"])

    summary = (
        split_df.groupby(["split", "collection_name"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "collection_name"])
        .reset_index(drop=True)
    )
    return summary
