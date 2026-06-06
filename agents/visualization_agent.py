"""
Visualization Agent — intelligent chart selection and Plotly rendering.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_COLORS = [
    "#6366f1", "#22d3ee", "#f59e0b", "#10b981",
    "#ef4444", "#8b5cf6", "#f97316", "#06b6d4",
]

CHART_OPTIONS = [
    "auto", "bar", "horizontal_bar", "line", "scatter",
    "histogram", "pie", "heatmap", "box", "area", "table",
]

PLOTLY_TEMPLATE = "plotly_dark"


def select_chart_type(
    result: pd.DataFrame,
    context: Optional[dict[str, Any]] = None,
    profile: Optional[dict[str, Any]] = None,
) -> str:
    """
    Intelligent chart selection based on result shape, intent, and column types.
    Never blindly plots first numeric columns without reasoning.
    """
    if result is None or result.empty:
        return "table"

    context = context or {}
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()
    intent = context.get("intent_type", "")
    recommended = context.get("recommended_chart", "")

    if recommended and recommended in CHART_OPTIONS and recommended != "auto":
        return _validate_chart_type(result, recommended, numeric_cols, categorical_cols)

    if intent == "trend_analysis" or (profile and profile.get("datetime_columns")):
        if len(result) >= 3 and numeric_cols:
            return "line"

    if intent == "statistical_analysis" and len(numeric_cols) >= 2:
        return "scatter"

    if intent in ("aggregation", "comparison", "ranking"):
        if len(result) <= 15 and categorical_cols and numeric_cols:
            return "horizontal_bar" if len(result) > 8 else "bar"
        if categorical_cols and numeric_cols:
            return "bar"

    if intent == "data_retrieval":
        if len(numeric_cols) == 1 and len(categorical_cols) == 0:
            return "histogram"
        if categorical_cols and numeric_cols:
            return "bar"
        return "table"

    if intent == "visualization":
        if len(numeric_cols) == 1:
            return "histogram"
        if len(numeric_cols) >= 2:
            return "scatter"

    if len(numeric_cols) >= 2 and len(categorical_cols) == 0:
        if len(result) > 20:
            return "scatter"
        return "heatmap"

    if categorical_cols and numeric_cols:
        return "horizontal_bar" if len(result) > 10 else "bar"

    if len(numeric_cols) == 1:
        return "histogram"

    if categorical_cols:
        return "bar"

    return "table"


def create_visualization(
    result: pd.DataFrame,
    chart_type: str,
    question: str,
    profile: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> tuple[Optional[go.Figure], str, Optional[str]]:
    """
    Create chart. Returns (figure, resolved_chart_type, error_message).
    """
    if result is None or result.empty:
        return None, "table", "No data to visualize."

    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()

    if chart_type == "auto":
        chart_type = select_chart_type(result, context, profile)

    chart_type = _validate_chart_type(result, chart_type, numeric_cols, categorical_cols)

    if chart_type == "table":
        return _create_table(result, question), "table", None

    if chart_type in ("bar", "horizontal_bar", "line", "scatter", "histogram", "pie", "heatmap", "box", "area"):
        if chart_type in ("bar", "horizontal_bar", "scatter", "heatmap", "box") and not numeric_cols:
            return None, chart_type, "Dataset contains no numeric columns for this chart type."

    try:
        fig = _build_chart(result, chart_type, numeric_cols, categorical_cols, question)
        if fig:
            fig = _apply_theme(fig)
        return fig, chart_type, None
    except Exception as e:
        return _create_table(result, question), "table", f"Chart rendering failed: {e}"


def _validate_chart_type(df, chart_type, numeric_cols, categorical_cols) -> str:
    if chart_type == "scatter" and len(numeric_cols) < 2:
        return "bar" if categorical_cols else "histogram"
    if chart_type in ("line", "area") and len(df) < 3:
        return "bar"
    if chart_type == "pie" and (not categorical_cols or not numeric_cols):
        return "bar"
    if chart_type == "heatmap" and len(numeric_cols) < 2:
        return "bar" if numeric_cols else "table"
    if chart_type == "box" and not numeric_cols:
        return "bar" if categorical_cols else "table"
    return chart_type


def _build_chart(df, chart_type, numeric_cols, categorical_cols, title):
    builders = {
        "bar": _bar_chart,
        "horizontal_bar": _horizontal_bar_chart,
        "line": _line_chart,
        "scatter": _scatter_chart,
        "histogram": _histogram,
        "pie": _pie_chart,
        "heatmap": _heatmap,
        "box": _box_plot,
        "area": _area_chart,
    }
    return builders.get(chart_type, _bar_chart)(df, numeric_cols, categorical_cols, title)


def _pick_axes(df, numeric_cols, categorical_cols):
    x_cat = categorical_cols[0] if categorical_cols else None
    y_num = numeric_cols[0] if numeric_cols else None
    x_num = numeric_cols[0] if numeric_cols else None
    y_num2 = numeric_cols[1] if len(numeric_cols) > 1 else None
    return x_cat, y_num, x_num, y_num2


def _bar_chart(df, numeric_cols, categorical_cols, title):
    x_cat, y_num, _, _ = _pick_axes(df, numeric_cols, categorical_cols)
    if not y_num:
        col = categorical_cols[0] if categorical_cols else df.columns[0]
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        return px.bar(counts.head(20), x=col, y="count", title=title, color_discrete_sequence=CHART_COLORS)
    x_col = x_cat or df.columns[0]
    df_sorted = df.sort_values(y_num, ascending=False).head(25)
    return px.bar(df_sorted, x=x_col, y=y_num, title=title, color_discrete_sequence=CHART_COLORS, text=y_num)


def _horizontal_bar_chart(df, numeric_cols, categorical_cols, title):
    x_cat, y_num, _, _ = _pick_axes(df, numeric_cols, categorical_cols)
    if not y_num or not x_cat:
        return _bar_chart(df, numeric_cols, categorical_cols, title)
    df_sorted = df.sort_values(y_num, ascending=True).tail(20)
    return px.bar(df_sorted, x=y_num, y=x_cat, orientation="h", title=title, color_discrete_sequence=CHART_COLORS)


def _line_chart(df, numeric_cols, categorical_cols, title):
    x_cat, y_num, _, _ = _pick_axes(df, numeric_cols, categorical_cols)
    work = df.copy()
    if x_cat and x_cat in work.columns:
        x_col = x_cat
    else:
        work = work.reset_index()
        x_col = work.columns[0]
    y_cols = numeric_cols[:3]
    return px.line(work, x=x_col, y=y_cols[0] if len(y_cols) == 1 else y_cols,
                   title=title, color_discrete_sequence=CHART_COLORS, markers=True)


def _scatter_chart(df, numeric_cols, categorical_cols, title):
    if len(numeric_cols) < 2:
        return _bar_chart(df, numeric_cols, categorical_cols, title)
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    color_col = categorical_cols[0] if categorical_cols else None
    return px.scatter(df, x=x_col, y=y_col, color=color_col, title=title,
                      color_discrete_sequence=CHART_COLORS,
                      trendline="ols" if len(df) > 10 else None)


def _histogram(df, numeric_cols, categorical_cols, title):
    if not numeric_cols:
        col = categorical_cols[0] if categorical_cols else df.columns[0]
        return px.histogram(df, x=col, title=title, color_discrete_sequence=CHART_COLORS)
    return px.histogram(df, x=numeric_cols[0], title=title, nbins=min(40, max(10, len(df) // 5)),
                        color_discrete_sequence=CHART_COLORS, marginal="box")


def _pie_chart(df, numeric_cols, categorical_cols, title):
    if not categorical_cols or not numeric_cols:
        return _bar_chart(df, numeric_cols, categorical_cols, title)
    df_top = df.nlargest(8, numeric_cols[0]) if len(df) > 8 else df
    return px.pie(df_top, names=categorical_cols[0], values=numeric_cols[0], title=title,
                  color_discrete_sequence=CHART_COLORS, hole=0.35)


def _heatmap(df, numeric_cols, categorical_cols, title):
    if len(numeric_cols) < 2:
        return _bar_chart(df, numeric_cols, categorical_cols, title)
    return px.imshow(df[numeric_cols].corr(), title=title, color_continuous_scale="RdBu_r", aspect="auto", text_auto=".2f")


def _box_plot(df, numeric_cols, categorical_cols, title):
    y_col = numeric_cols[0]
    x_col = categorical_cols[0] if categorical_cols else None
    return px.box(df, x=x_col, y=y_col, title=title, color=x_col, color_discrete_sequence=CHART_COLORS, points="outliers")


def _area_chart(df, numeric_cols, categorical_cols, title):
    work = df.copy()
    x_cat, y_num, _, _ = _pick_axes(df, numeric_cols, categorical_cols)
    if x_cat and x_cat in work.columns:
        x_col = x_cat
    else:
        work = work.reset_index()
        x_col = work.columns[0]
    y_col = y_num or work.columns[-1]
    return px.area(work, x=x_col, y=y_col, title=title, color_discrete_sequence=CHART_COLORS)


def _create_table(df, title):
    display = df.head(50)
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in display.columns],
                    fill_color="#1e1e2e", font=dict(color="white", size=12), align="left"),
        cells=dict(values=[display[col].tolist() for col in display.columns],
                   fill_color=["#13131f", "#1a1a2e"], font=dict(color="#c0c0d0", size=11), align="left"),
    )])
    fig.update_layout(title=title)
    return fig


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.8)",
        font=dict(family="Inter, sans-serif", color="#c0c0d0"),
        title=dict(font=dict(size=15, color="#e2e2f0"), x=0.05),
        legend=dict(bgcolor="rgba(20,20,35,0.8)", bordercolor="rgba(100,100,150,0.3)", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
        colorway=CHART_COLORS,
    )
    return fig


def create_kpi_dashboard(profile: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_rows": f"{profile.get('rows', 0):,}",
        "total_columns": str(profile.get("columns", 0)),
        "missing_pct": f"{profile.get('total_missing_pct', 0):.1f}%",
        "quality_score": f"{profile.get('quality_score', 0)}/100",
        "duplicate_rows": f"{profile.get('duplicate_rows', 0):,}",
        "dataset_type": profile.get("structure_type", "Unknown"),
    }


def fig_to_bytes(fig: go.Figure) -> bytes:
    try:
        return fig.to_image(format="png", width=800, height=400, scale=2)
    except Exception:
        plt_fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f14")
        ax.text(0.5, 0.5, "Chart export unavailable", ha="center", va="center", color="white", fontsize=14)
        ax.set_facecolor("#0f0f14")
        buf = io.BytesIO()
        plt_fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f0f14")
        plt.close(plt_fig)
        return buf.getvalue()
