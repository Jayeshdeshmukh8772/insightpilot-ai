"""
Execution Engine - Safely runs DuckDB SQL and Pandas code
Agent 4: Execution Planning + Code Runner
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Dict, Any, Tuple, Optional
import traceback
import re


def execute_query(
    df: pd.DataFrame,
    code: str,
    code_type: str = "sql",
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute SQL (DuckDB) or Pandas code against the dataframe.
    Returns (result_df, error_message)
    """
    if code_type == "sql":
        return _execute_sql(df, code)
    else:
        return _execute_pandas(df, code)


def _execute_sql(df: pd.DataFrame, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute DuckDB SQL against the dataframe."""
    try:
        sql = _sanitize_sql(sql)
        conn = duckdb.connect(database=":memory:")
        conn.register("df", df)
        result = conn.execute(sql).fetchdf()
        conn.close()
        return result, None
    except Exception as e:
        error_msg = str(e)
        # Try to recover with simplified query
        try:
            simplified = _simplify_sql(sql, df)
            if simplified and simplified != sql:
                conn2 = duckdb.connect(database=":memory:")
                conn2.register("df", df)
                result = conn2.execute(simplified).fetchdf()
                conn2.close()
                return result, f"⚠️ Query simplified: {error_msg}"
        except Exception:
            pass
        return None, f"SQL Error: {error_msg}"


def _execute_pandas(df: pd.DataFrame, code: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute Pandas code in a sandboxed namespace."""
    try:
        code = _sanitize_pandas_code(code)
        namespace = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "result": None,
        }
        exec(code, namespace)
        result = namespace.get("result")
        
        if result is None:
            return None, "Code executed but no 'result' variable was set."
        
        if isinstance(result, pd.Series):
            result = result.reset_index()
            result.columns = [str(c) for c in result.columns]
        
        if isinstance(result, pd.DataFrame):
            # Ensure clean column names
            result.columns = [str(c) for c in result.columns]
            return result, None
        
        # Scalar result
        result_df = pd.DataFrame({"Result": [result]})
        return result_df, None
        
    except Exception as e:
        return None, f"Execution Error: {str(e)}\n{traceback.format_exc()[-500:]}"


def _sanitize_sql(sql: str) -> str:
    """Basic SQL safety checks."""
    sql = sql.strip().rstrip(";")
    # Block destructive operations
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"]
    upper_sql = sql.upper()
    for keyword in dangerous:
        if re.search(rf'\b{keyword}\b', upper_sql):
            raise ValueError(f"Disallowed SQL operation: {keyword}")
    return sql


def _sanitize_pandas_code(code: str) -> str:
    """Basic Python safety checks for pandas code."""
    blocked = [
        "import os", "import sys", "import subprocess", "subprocess",
        "eval(", "exec(", "__import__", "open(", "compile(",
        "getattr(", "__builtins__", "globals(", "locals(",
    ]
    lower = code.lower()
    for b in blocked:
        if b.lower() in lower:
            raise ValueError(f"Disallowed operation in code: {b}")
    return code


def _simplify_sql(sql: str, df: pd.DataFrame) -> Optional[str]:
    """Attempt to fix common SQL issues."""
    # If column not found, try case-insensitive match
    col_map = {col.lower(): col for col in df.columns}
    for col_lower, col_actual in col_map.items():
        sql = re.sub(rf'\b{col_lower}\b', col_actual, sql, flags=re.IGNORECASE)
    return sql


def get_result_summary(result: pd.DataFrame, max_rows: int = 5) -> str:
    """Create a text summary of query results for insight generation."""
    if result is None or result.empty:
        return "No results returned."
    
    lines = [f"Shape: {result.shape[0]} rows × {result.shape[1]} columns"]
    
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:3]:
            stats = result[col].describe()
            lines.append(
                f"Column '{col}': min={stats.get('min', 'N/A'):.2f}, "
                f"max={stats.get('max', 'N/A'):.2f}, "
                f"mean={stats.get('mean', 'N/A'):.2f}"
            )
    
    lines.append(f"\nTop {min(max_rows, len(result))} rows:")
    lines.append(result.head(max_rows).to_string(index=False))
    
    return "\n".join(lines)