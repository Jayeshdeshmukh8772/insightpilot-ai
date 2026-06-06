"""
Schema analysis utilities — local-only, no LLM dependency.
Works on arbitrary tabular schemas without domain assumptions.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


ID_KEYWORDS = frozenset({"id", "index", "key", "code", "no", "num", "seq", "uuid", "guid"})


def is_likely_id_column(col_name: str, unique_pct: float) -> bool:
    """Heuristic: column name or near-unique values suggest an identifier."""
    col_lower = col_name.lower().strip()
    if col_lower in ID_KEYWORDS:
        return True
    for kw in ID_KEYWORDS:
        if col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_"):
            return True
    return unique_pct > 95 and col_lower.endswith("id")


def detect_datetime_columns(df: pd.DataFrame, categorical_cols: list[str]) -> list[str]:
    """Detect string columns that parse as datetimes."""
    detected: list[str] = []
    for col in categorical_cols:
        try:
            sample = df[col].dropna().head(30)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() / max(len(sample), 1) > 0.7:
                detected.append(col)
        except Exception:
            pass
    return detected


def count_outliers(series: pd.Series) -> int:
    """IQR-based outlier count."""
    try:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0
        return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    except Exception:
        return 0


def compute_quality_score(df: pd.DataFrame, missing_pct: pd.Series, duplicate_rows: int) -> int:
    """Compute a 0-100 data quality score."""
    score = 100.0
    avg_missing = float(missing_pct.mean()) if len(missing_pct) else 0.0
    score -= min(40, avg_missing * 2)
    dup_pct = duplicate_rows / max(len(df), 1) * 100
    score -= min(20, dup_pct * 2)
    if len(df) < 50:
        score -= 10
    return max(0, int(score))


def build_column_profile(df: pd.DataFrame, col: str, numeric_cols: list[str], detected_datetime: list[str]) -> dict[str, Any]:
    """Build a single column profile dict."""
    col_data = df[col]
    dtype = str(col_data.dtype)
    null_count = int(col_data.isnull().sum())
    unique_count = int(col_data.nunique())
    unique_pct = round(unique_count / max(len(df), 1) * 100, 2)

    profile: dict[str, Any] = {
        "name": col,
        "dtype": dtype,
        "null_count": null_count,
        "null_pct": round(null_count / max(len(df), 1) * 100, 2),
        "unique_count": unique_count,
        "unique_pct": unique_pct,
        "is_likely_id": is_likely_id_column(col, unique_pct),
    }

    if col in numeric_cols:
        profile["type"] = "numeric"
        for stat in ("min", "max", "mean", "median", "std"):
            val = col_data.min() if stat == "min" else col_data.max() if stat == "max" else getattr(col_data, stat)()
            profile[stat] = round(float(val), 4) if pd.notna(val) else None
        skew = col_data.skew()
        profile["skewness"] = round(float(skew), 4) if pd.notna(skew) else None
        profile["outlier_count"] = count_outliers(col_data)
    elif col in detected_datetime:
        profile["type"] = "datetime"
        profile["sample_values"] = col_data.dropna().head(3).astype(str).tolist()
    else:
        top_vals = col_data.value_counts().head(5)
        profile["type"] = "categorical"
        profile["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        profile["sample_values"] = col_data.dropna().head(3).astype(str).tolist()

    return profile


def infer_structure_type(column_profiles: list[dict[str, Any]]) -> str:
    """Schema-agnostic structure label based on column types only."""
    if not column_profiles:
        return "Empty Dataset"
    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0}
    for cp in column_profiles:
        t = cp.get("type", "categorical")
        if t in type_counts:
            type_counts[t] += 1
    total = len(column_profiles)
    numeric_ratio = type_counts["numeric"] / total
    if type_counts["datetime"] >= 1 and type_counts["numeric"] >= 1:
        return "Time-Series Tabular Data"
    if numeric_ratio > 0.7:
        return "Numeric-Heavy Tabular Data"
    if numeric_ratio < 0.25:
        return "Categorical-Heavy Tabular Data"
    return "Mixed Tabular Data"


def analysis_ready_columns(profile: dict[str, Any]) -> dict[str, list[str]]:
    """Return columns suitable for analysis (excluding likely IDs)."""
    numeric = [
        c["name"] for c in profile.get("column_profiles", [])
        if c.get("type") == "numeric" and not c.get("is_likely_id")
    ]
    categorical = [
        c["name"] for c in profile.get("column_profiles", [])
        if c.get("type") == "categorical" and not c.get("is_likely_id")
    ]
    datetime_cols = [c["name"] for c in profile.get("column_profiles", []) if c.get("type") == "datetime"]

    if not numeric:
        numeric = profile.get("numeric_columns", [])
    if not categorical:
        categorical = profile.get("categorical_columns", [])

    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}
