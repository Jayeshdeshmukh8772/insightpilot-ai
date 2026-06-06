"""
Data Loader Utility
Handles CSV, Excel, JSON, Parquet, TXT with smart parsing and type coercion.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {
    "csv": "CSV File",
    "txt": "Text File (CSV)",
    "xlsx": "Excel Workbook",
    "xls": "Excel Legacy",
    "json": "JSON File",
    "parquet": "Parquet File",
    "pq": "Parquet File",
}


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load an uploaded Streamlit file into a pandas DataFrame.
    Returns (dataframe, error_message).
    """
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in SUPPORTED_FORMATS:
        return None, (
            f"Unsupported file format: .{ext}. "
            f"Supported: {', '.join(sorted(set(SUPPORTED_FORMATS.keys())))}"
        )

    try:
        content = uploaded_file.read()

        if ext in ("csv", "txt"):
            df, err = _load_csv(content)
        elif ext in ("xlsx", "xls"):
            df, err = _load_excel(content)
        elif ext == "json":
            df, err = _load_json(content)
        elif ext in ("parquet", "pq"):
            df, err = _load_parquet(content)
        else:
            return None, f"Handler not implemented for .{ext}"

        if err:
            return None, err

        df = _post_process(df)
        return df, None

    except Exception as e:
        return None, f"Failed to load file: {str(e)}"


def load_path(file_path: str | Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load a dataset from a local file path."""
    path = Path(file_path)
    if not path.exists():
        return None, f"File not found: {path}"

    ext = path.suffix.lstrip(".").lower()
    try:
        content = path.read_bytes()
        if ext in ("csv", "txt"):
            df, err = _load_csv(content)
        elif ext in ("xlsx", "xls"):
            df, err = _load_excel(content)
        elif ext == "json":
            df, err = _load_json(content)
        elif ext in ("parquet", "pq"):
            df, err = _load_parquet(content)
        else:
            return None, f"Unsupported format: .{ext}"

        if err:
            return None, err
        return _post_process(df), None
    except Exception as e:
        return None, f"Failed to load {path.name}: {str(e)}"


def _load_csv(content: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]
    separators = [",", ";", "\t", "|"]

    for encoding in encodings:
        try:
            text = content.decode(encoding)
            first_line = text.split("\n")[0] if text else ""
            for sep in separators:
                if sep in first_line:
                    df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
                    if df.shape[1] >= 1:
                        return df, None
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            return df, None
        except (UnicodeDecodeError, Exception):
            continue

    return None, "Could not decode file. Try saving as UTF-8 CSV."


def _load_excel(content: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    xl = pd.ExcelFile(io.BytesIO(content))
    best_df = None
    best_size = 0

    for sheet in xl.sheet_names[:5]:
        try:
            df = xl.parse(sheet)
            if df.size > best_size:
                best_df = df
                best_size = df.size
        except Exception:
            continue

    if best_df is None:
        return None, "No readable sheets found in Excel file."
    return best_df, None


def _load_json(content: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            for key in ("data", "records", "results", "items", "rows"):
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    break
            else:
                df = pd.DataFrame([data])
        else:
            return None, "JSON structure not supported. Provide a list of records."
        return df, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {str(e)}"


def _load_parquet(content: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        return pd.read_parquet(io.BytesIO(content)), None
    except Exception as e:
        return None, f"Parquet read error: {str(e)}"


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    for col in df.select_dtypes(include=["object"]).columns:
        try:
            cleaned = df[col].dropna().astype(str).str.replace(",", "", regex=False)
            cleaned = cleaned.str.replace("$", "", regex=False).str.replace("%", "", regex=False)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() / max(len(df[col].dropna()), 1) > 0.8:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("%", "", regex=False),
                    errors="coerce",
                )
        except Exception:
            pass

    df.columns = [
        str(c).strip().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        for c in df.columns
    ]
    return df
