"""
Caching layer for InsightPilot — reduces repeated computation and API calls.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import pandas as pd
import streamlit as st

from agents.dataset_agent import profile_dataset


def df_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash for a dataframe shape + column metadata."""
    meta = {
        "rows": len(df),
        "cols": list(df.columns.astype(str)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "head_hash": hashlib.md5(
            df.head(50).to_csv(index=False).encode("utf-8", errors="ignore")
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_profile(df_fingerprint: str, df_bytes: bytes, filename: str) -> dict[str, Any]:
    """Cache dataset profiling by fingerprint."""
    import io
    df = pd.read_parquet(io.BytesIO(df_bytes)) if df_bytes[:4] == b"PAR1" else pd.read_pickle(io.BytesIO(df_bytes))
    return profile_dataset(df, filename)


def get_or_build_profile(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Profile with caching — serializes df for cache key."""
    import io
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
        df_bytes = buf.getvalue()
    except Exception:
        buf = io.BytesIO()
        df.to_pickle(buf)
        df_bytes = buf.getvalue()

    fingerprint = df_fingerprint(df)
    return cached_profile(fingerprint, df_bytes, filename)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_query_result(query_hash: str, df_fingerprint: str, code: str, code_type: str, df_bytes: bytes):
    """Cache query execution results."""
    import io
    from agents.execution_agent import execute_query

    try:
        df = pd.read_parquet(io.BytesIO(df_bytes))
    except Exception:
        df = pd.read_pickle(io.BytesIO(df_bytes))

    return execute_query(df, code, code_type)


def query_cache_key(code: str, code_type: str, df_fingerprint: str) -> str:
    return hashlib.sha256(f"{df_fingerprint}:{code_type}:{code}".encode()).hexdigest()


def serialize_df_for_cache(df: pd.DataFrame) -> bytes:
    import io
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    except Exception:
        buf = io.BytesIO()
        df.to_pickle(buf)
        return buf.getvalue()


def get_provider_status() -> dict[str, Any]:
    """Cached provider connectivity check (resource-level)."""
    return _provider_status_impl()


@st.cache_resource(show_spinner=False, ttl=300)
def _provider_status_impl() -> dict[str, Any]:
    from agents.llm_agent import get_llm_config, is_configured

    cfg = get_llm_config()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "configured": is_configured(),
    }
