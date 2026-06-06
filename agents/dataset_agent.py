"""
Dataset Agent — local-only profiling and schema-aware suggested questions.
No LLM calls. Works on arbitrary schemas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.schema_analyzer import (
    analysis_ready_columns,
    build_column_profile,
    compute_quality_score,
    detect_datetime_columns,
    infer_structure_type,
)


def profile_dataset(df: pd.DataFrame, filename: str = "dataset") -> dict[str, Any]:
    """Full statistical and structural profile — 100% local computation."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    detected_datetime = detect_datetime_columns(df, list(categorical_cols))
    categorical_cols = [c for c in categorical_cols if c not in detected_datetime]

    missing = df.isnull().sum()
    missing_pct = (missing / max(len(df), 1) * 100).round(2)
    missing_cols = missing[missing > 0].index.tolist()
    duplicate_rows = int(df.duplicated().sum())

    column_profiles = [
        build_column_profile(df, col, numeric_cols, detected_datetime)
        for col in df.columns
    ]

    quality_score = compute_quality_score(df, missing_pct, duplicate_rows)

    profile: dict[str, Any] = {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": list(datetime_cols) + detected_datetime,
        "boolean_columns": bool_cols,
        "missing_columns": missing_cols,
        "total_missing_pct": round(float(missing_pct.mean()), 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / max(len(df), 1) * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        "quality_score": quality_score,
        "column_profiles": column_profiles,
        "structure_type": infer_structure_type(column_profiles),
    }
    return profile


def generate_suggested_questions(profile: dict[str, Any]) -> list[str]:
    """Schema-driven question suggestions using actual column names only."""
    cols = analysis_ready_columns(profile)
    numeric = cols["numeric"]
    categorical = cols["categorical"]
    datetime_cols = cols["datetime"]
    questions: list[str] = []

    if categorical and numeric:
        questions.append(f"What is the average {numeric[0]} grouped by {categorical[0]}?")
        questions.append(f"Show top 10 {categorical[0]} values by total {numeric[0]}")

    if len(categorical) >= 1:
        cat = categorical[0]
        questions.append(f"How many records exist for each {cat}?")
        if len(categorical) >= 2:
            questions.append(
                f"Filter records where {categorical[0]} equals the most common {categorical[0]} value"
            )

    if len(numeric) >= 2:
        questions.append(f"What is the correlation between {numeric[0]} and {numeric[1]}?")

    if numeric:
        questions.append(f"Show the distribution of {numeric[0]}")
        questions.append(f"What are the highest and lowest values of {numeric[0]}?")

    if datetime_cols and numeric:
        questions.append(f"How does {numeric[0]} change over {datetime_cols[0]}?")

    if profile.get("total_missing_pct", 0) > 1:
        questions.append("Which columns have missing values and how many?")

    if profile.get("duplicate_rows", 0) > 0:
        questions.append("How many duplicate rows exist in this dataset?")

    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:8]


def build_local_dataset_summary(profile: dict[str, Any]) -> str:
    """Local text summary — no LLM required for dataset overview."""
    cols = analysis_ready_columns(profile)
    return (
        f"**{profile.get('filename', 'Dataset')}** contains {profile.get('rows', 0):,} rows "
        f"and {profile.get('columns', 0)} columns ({profile.get('structure_type', 'tabular')}). "
        f"Numeric columns: {len(cols['numeric'])} | Categorical: {len(cols['categorical'])} | "
        f"Datetime: {len(cols['datetime'])}. "
        f"Data quality score: **{profile.get('quality_score', 0)}/100** "
        f"({profile.get('total_missing_pct', 0):.1f}% missing, "
        f"{profile.get('duplicate_rows', 0):,} duplicates)."
    )
