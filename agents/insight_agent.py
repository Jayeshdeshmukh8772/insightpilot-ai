"""
Insight Agent — optional LLM-powered insights, chart explanations, executive summaries.
All functions are on-demand (button-triggered only).
"""

from __future__ import annotations

from typing import Any

from agents.llm_agent import LLMError, call_llm


def generate_insights(
    question: str,
    result_summary: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Generate analytical insights from query results."""
    prompt = f"""You are a senior data analyst.

Dataset structure: {profile.get('structure_type', 'tabular')} | {profile.get('rows', 0):,} rows
User question: "{question}"

Query result summary:
{result_summary}

Generate data-driven insights using ACTUAL numbers from the result summary.
Do not invent data not present in the summary.

Return ONLY valid JSON:
{{
  "key_findings": ["finding with specific numbers", "finding 2", "finding 3"],
  "pattern": "identified pattern or trend",
  "anomaly": "anomaly or risk if any, else null",
  "recommendation": "one actionable recommendation",
  "confidence_level": "High|Medium|Low"
}}"""

    result = call_llm(prompt, json_mode=True)
    if not result.get("key_findings"):
        raise LLMError("Insight generation failed: no findings returned.")
    return result


def explain_chart(
    chart_type: str,
    question: str,
    result_summary: str,
    profile: dict[str, Any],
) -> str:
    """Plain-English chart explanation for learners."""
    prompt = f"""You are a data storytelling teacher explaining a chart to a student.

Chart type: {chart_type}
Question analyzed: "{question}"
Dataset: {profile.get('structure_type', 'tabular')} with {profile.get('rows', 0):,} rows

Data behind the chart:
{result_summary[:2000]}

In 2-3 clear sentences explain:
1. What this chart shows
2. The most important takeaway (use real values if available)
3. What follow-up question a student should ask

Be specific. No generic filler."""

    text = call_llm(prompt, json_mode=False)
    if not text or len(text) < 20:
        raise LLMError("Chart explanation generation failed.")
    return str(text)


def generate_executive_summary(
    profile: dict[str, Any],
    qa_history: list[dict[str, Any]],
) -> str:
    """C-level executive summary — on-demand only."""
    qa_text = []
    for item in qa_history[-5:]:
        findings = item.get("insight", {}).get("key_findings", [])
        finding = findings[0] if findings else "No insights generated yet"
        qa_text.append(f"Q: {item.get('question', '')}\nFinding: {finding}")

    prompt = f"""Write a concise executive summary for a data analysis session.

Dataset: {profile.get('filename', 'dataset')}
Structure: {profile.get('structure_type', 'tabular')}
Rows: {profile.get('rows', 0):,} | Quality score: {profile.get('quality_score', 0)}/100

Completed analyses:
{chr(10).join(qa_text) if qa_text else 'No analyses completed yet.'}

Write 3 short paragraphs:
1. Dataset overview and quality
2. Key findings from completed analyses
3. Strategic recommendations

Use specific metrics. Professional tone."""

    text = call_llm(prompt, json_mode=False)
    if not text or len(text) < 50:
        raise LLMError("Executive summary generation failed.")
    return str(text)
