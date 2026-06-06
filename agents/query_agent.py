"""
Query Agent — question understanding and SQL/Pandas generation.
Two explicit LLM calls, triggered only by user action.
"""

from __future__ import annotations

from typing import Any

from agents.llm_agent import LLMError, call_llm


def _column_catalog(profile: dict[str, Any]) -> str:
    lines = []
    for col in profile.get("column_profiles", []):
        extras = []
        if col.get("type") == "numeric":
            extras.append(f"range={col.get('min')}..{col.get('max')}")
        elif col.get("type") == "categorical":
            top = col.get("top_values", {})
            if top:
                extras.append(f"top_values={list(top.keys())[:5]}")
        elif col.get("type") == "datetime":
            extras.append(f"samples={col.get('sample_values', [])}")
        line = f"  - {col['name']} ({col.get('type', 'unknown')}, dtype={col.get('dtype')})"
        if extras:
            line += " [" + ", ".join(extras) + "]"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no columns)"


def understand_question(question: str, profile: dict[str, Any]) -> dict[str, Any]:
    """
    Step 2: Single LLM call — intent, enhanced question, chart hint, execution engine.
    """
    catalog = _column_catalog(profile)
    prompt = f"""You are a data analytics teaching assistant. Analyze the user's question against the ACTUAL dataset schema.

Dataset: {profile.get('filename', 'dataset')}
Rows: {profile.get('rows', 0):,} | Columns: {profile.get('columns', 0)}
Structure: {profile.get('structure_type', 'tabular')}

ACTUAL COLUMNS (use ONLY these exact names):
{catalog}

User question: "{question}"

Rules:
- NEVER invent columns that are not listed above
- Map the question to real column names from the schema
- Filter questions need WHERE conditions on matching columns
- Aggregation questions need GROUP BY and aggregate functions
- Ranking questions need ORDER BY and LIMIT
- Different questions MUST require different logic

Return ONLY valid JSON:
{{
  "intent_type": "data_retrieval|aggregation|visualization|statistical_analysis|trend_analysis|comparison|ranking|data_quality",
  "confidence": 0.0,
  "enhanced_question": "precise analytical question using exact column names",
  "analysis_method": "SQL Filter|SQL Aggregation|Pandas Transform|Time Series|Correlation",
  "recommended_chart": "bar|horizontal_bar|line|scatter|histogram|pie|heatmap|box|table",
  "execution_approach": "duckdb|pandas",
  "key_columns": ["col1", "col2"],
  "needs_filter": true,
  "needs_aggregation": false,
  "reasoning": "brief explanation of analytical approach"
}}"""

    result = call_llm(prompt, json_mode=True)
    required = ("intent_type", "enhanced_question", "execution_approach", "recommended_chart")
    missing = [k for k in required if k not in result]
    if missing:
        raise LLMError(f"Question understanding incomplete. Missing fields: {missing}")
    return result


def generate_query(
    question: str,
    profile: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Step 3: Single LLM call — generate executable DuckDB SQL or Pandas code.
    """
    catalog = _column_catalog(profile)
    approach = context.get("execution_approach", "duckdb")
    enhanced = context.get("enhanced_question", question)
    intent = context.get("intent_type", "analysis")

    if approach == "duckdb":
        prompt = f"""You are a DuckDB SQL expert teaching analytics.

Table name: df (already loaded)
ACTUAL COLUMNS:
{catalog}

Intent: {intent}
Question: "{enhanced}"

Write DuckDB SQL that answers THIS specific question — not generic statistics.

Examples of question-specific logic:
- "Find records where city = 'Pune'" → SELECT * FROM df WHERE city = 'Pune' LIMIT 500
- "Which team won most tosses?" → SELECT team, COUNT(*) AS toss_wins FROM df WHERE toss_winner = team GROUP BY team ORDER BY toss_wins DESC LIMIT 10

Rules:
- Use ONLY column names from the schema above (case-sensitive)
- Use TRY_CAST for type safety
- Add LIMIT 500 for row-level SELECT *
- For text matching use ILIKE or exact match as appropriate
- NEVER use df.describe() or generic summaries
- Query must directly answer the question

Return ONLY valid JSON:
{{
  "code_type": "sql",
  "code": "SELECT ...",
  "explanation": "what this query does and why"
}}"""
    else:
        prompt = f"""You are a pandas expert teaching analytics.

DataFrame variable: df (already loaded)
ACTUAL COLUMNS:
{catalog}

Intent: {intent}
Question: "{enhanced}"

Write pandas code that answers THIS specific question — not generic statistics.

Rules:
- Store final answer in variable `result` (DataFrame or Series)
- pd and np are pre-imported — do not import anything
- Use ONLY column names from the schema above
- NEVER use df.describe() as a fallback
- Code must directly answer the question

Return ONLY valid JSON:
{{
  "code_type": "pandas",
  "code": "result = df[...]",
  "explanation": "what this code does and why"
}}"""

    result = call_llm(prompt, json_mode=True)
    if "code" not in result or not str(result.get("code", "")).strip():
        raise LLMError("Query generation failed: LLM returned empty code.")

    code = str(result["code"]).strip()
    if "describe()" in code.lower() and "describe" not in question.lower():
        raise LLMError(
            "Query generation produced a generic describe() fallback. "
            "Rephrase your question or try again."
        )

    result["code_type"] = result.get("code_type", "sql" if approach == "duckdb" else "pandas")
    return result
