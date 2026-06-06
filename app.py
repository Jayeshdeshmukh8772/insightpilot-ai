"""
InsightPilot — Universal AI-Powered Data Analytics Platform
User-controlled LLM workflow with local profiling and execution.
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils.data_loader import load_file, load_path
from utils.cache_manager import (
    get_or_build_profile,
    get_provider_status,
    query_cache_key,
    serialize_df_for_cache,
    cached_query_result,
    df_fingerprint,
)
from agents.dataset_agent import generate_suggested_questions, build_local_dataset_summary
from agents.query_agent import understand_question, generate_query
from agents.execution_agent import execute_query, get_result_summary
from agents.visualization_agent import (
    CHART_OPTIONS,
    create_visualization,
    create_kpi_dashboard,
    fig_to_bytes,
)
from agents.insight_agent import generate_insights, explain_chart, generate_executive_summary
from agents.llm_agent import LLMError, get_llm_config, is_configured
from agents.report_agent import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightPilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --indigo:#6366f1; --cyan:#22d3ee; --green:#10b981; --amber:#f59e0b; --red:#ef4444;
    --bg:#0b0b10; --surface:#14141f; --card:#1c1c2a; --text:#e4e4f0; --muted:#7878a0;
    --border:rgba(99,102,241,.18); --radius:10px;
}
html,body,.stApp { background:var(--bg)!important; font-family:'Inter',sans-serif; }
.main .block-container { padding:1rem 1.5rem 2rem; max-width:1440px; }
section[data-testid="stSidebar"] { background:#0e0e18!important; border-right:1px solid var(--border); }
#MainMenu,footer { visibility:hidden; height:0; } header[data-testid="stHeader"] { background: transparent;}
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-bottom:1rem; }
@media(max-width:900px){ .kpi-grid{ grid-template-columns:repeat(3,1fr); } }
@media(max-width:500px){ .kpi-grid{ grid-template-columns:repeat(2,1fr); } }
.kpi { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:.85rem; text-align:center; }
.kpi-val { font-size:1.3rem; font-weight:700; color:var(--indigo); }
.kpi-lbl { font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-size:.72rem; font-weight:600; }
.badge-ok { background:rgba(16,185,129,.12); color:var(--green); border:1px solid rgba(16,185,129,.25); }
.badge-warn { background:rgba(245,158,11,.12); color:var(--amber); border:1px solid rgba(245,158,11,.25); }
.badge-err { background:rgba(239,68,68,.12); color:var(--red); border:1px solid rgba(239,68,68,.25); }
.badge-info { background:rgba(99,102,241,.12); color:var(--indigo); border:1px solid rgba(99,102,241,.25); }
.panel { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1rem 1.2rem; margin-bottom:.75rem; }
.panel-title { font-size:.72rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }
.status-row { display:flex; flex-wrap:wrap; gap:8px; margin:.5rem 0; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:var(--card); border-radius:8px 8px 0 0; padding:8px 16px; }
.stButton>button { border-radius:var(--radius)!important; font-weight:600!important; }
.stTextInput>div>div>input, .stTextArea textarea { background:var(--card)!important; color:var(--text)!important; border-color:var(--border)!important; }
.query-box { font-family:ui-monospace,monospace; font-size:.82rem; background:#12121c; border:1px solid var(--border); border-radius:8px; padding:.75rem; color:var(--cyan); white-space:pre-wrap; }
.error-box { background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.3); border-radius:8px; padding:.75rem 1rem; color:#fca5a5; }
</style>
""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "df": None,
        "profile": None,
        "df_fingerprint": "",
        "df_cache_bytes": b"",
        "local_summary": "",
        "suggested_questions": [],
        "current_question": "",
        "question_context": None,
        "generated_query": None,
        "result_df": None,
        "exec_error": None,
        "chart": None,
        "chart_type": "auto",
        "chart_error": None,
        "insights": None,
        "chart_explanation": None,
        "exec_summary": "",
        "qa_history": [],
        "chart_images": [],
        "status": {},
        "last_error": None,
        "active_tab": 0,
        "_last_file": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _set_status(key: str, value: str, level: str = "info"):
    st.session_state.status[key] = {"value": value, "level": level, "at": datetime.now().strftime("%H:%M:%S")}


def _show_error(msg: str):
    st.session_state.last_error = msg
    st.markdown(f'<div class="error-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def _require_llm() -> bool:
    if not is_configured():
        cfg = get_llm_config()
        key_name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(
            cfg["provider"], "API key"
        )
        _show_error(f"{cfg['provider'].title()} API key not configured. Set {key_name} in .env or Streamlit secrets.")
        return False
    return True


def _load_dataset(df: pd.DataFrame, filename: str):
    st.session_state.df = df
    st.session_state.df_fingerprint = df_fingerprint(df)
    st.session_state.df_cache_bytes = serialize_df_for_cache(df)
    with st.spinner("Profiling dataset locally…"):
        st.session_state.profile = get_or_build_profile(df, filename)
    st.session_state.local_summary = build_local_dataset_summary(st.session_state.profile)
    st.session_state.suggested_questions = generate_suggested_questions(st.session_state.profile)
    st.session_state.qa_history = []
    st.session_state.chart_images = []
    st.session_state.exec_summary = ""
    st.session_state.current_question = ""
    st.session_state.question_context = None
    st.session_state.generated_query = None
    st.session_state.result_df = None
    st.session_state.exec_error = None
    st.session_state.chart = None
    st.session_state.insights = None
    st.session_state.chart_explanation = None
    st.session_state.last_error = None
    _set_status("dataset", f"Loaded {filename}", "ok")


def _render_kpis():
    if not st.session_state.profile:
        return
    kpis = create_kpi_dashboard(st.session_state.profile, st.session_state.df)
    items = [
        (kpis["total_rows"], "Rows"),
        (kpis["total_columns"], "Columns"),
        (kpis["quality_score"], "Quality"),
        (kpis["missing_pct"], "Missing"),
        (kpis["duplicate_rows"], "Duplicates"),
        (kpis["dataset_type"], "Structure"),
    ]
    html = '<div class="kpi-grid">'
    for val, lbl in items:
        html += f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_status_bar():
    cfg = get_llm_config()
    provider_cls = "badge-ok" if is_configured() else "badge-err"
    provider_lbl = f"✓ {cfg['provider'].title()} ({cfg['model']})" if is_configured() else f"✗ {cfg['provider'].title()} not configured"
    st.markdown(
        f'<div class="status-row">'
        f'<span class="badge {provider_cls}">{provider_lbl}</span>'
        f'<span class="badge badge-info">LLM calls: user-triggered only</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.last_error:
        _show_error(st.session_state.last_error)


def _handle_generate_query():
    q = st.session_state.current_question.strip()
    if not q:
        _show_error("Enter a question before generating a query.")
        return
    if not _require_llm():
        return
    profile = st.session_state.profile
    try:
        with st.spinner("Understanding question (1 LLM call)…"):
            ctx = understand_question(q, profile)
            st.session_state.question_context = ctx
            _set_status("understanding", ctx.get("intent_type", "done"), "ok")

        with st.spinner("Generating query (1 LLM call)…"):
            code_info = generate_query(q, profile, ctx)
            st.session_state.generated_query = code_info
            st.session_state.result_df = None
            st.session_state.exec_error = None
            st.session_state.chart = None
            st.session_state.insights = None
            _set_status("query", code_info.get("code_type", "sql"), "ok")
            st.session_state.last_error = None
    except LLMError as e:
        _show_error(str(e))
        _set_status("query", "failed", "error")


def _handle_run_query():
    code_info = st.session_state.generated_query
    if not code_info or not code_info.get("code"):
        _show_error("Generate a query first.")
        return
    df = st.session_state.df
    code = code_info["code"]
    code_type = code_info.get("code_type", "sql")
    try:
        with st.spinner("Executing query locally…"):
            qhash = query_cache_key(code, code_type, st.session_state.df_fingerprint)
            result_df, err = cached_query_result(
                qhash, st.session_state.df_fingerprint, code, code_type, st.session_state.df_cache_bytes
            )
            st.session_state.result_df = result_df
            st.session_state.exec_error = err
            st.session_state.chart = None
            st.session_state.insights = None
            if err and result_df is None:
                _show_error(err)
                _set_status("execution", "failed", "error")
            else:
                _set_status("execution", f"{len(result_df) if result_df is not None else 0} rows", "ok")
                st.session_state.last_error = None
    except Exception as e:
        _show_error(f"Query execution failed: {e}")


def _handle_generate_insights():
    if st.session_state.result_df is None or st.session_state.result_df.empty:
        _show_error("Run a query with results before generating insights.")
        return
    if not _require_llm():
        return
    try:
        with st.spinner("Generating insights (1 LLM call)…"):
            summary = get_result_summary(st.session_state.result_df)
            st.session_state.insights = generate_insights(
                st.session_state.current_question,
                summary,
                st.session_state.profile,
            )
            _set_status("insights", "generated", "ok")
            st.session_state.last_error = None
    except LLMError as e:
        _show_error(str(e))


def _handle_explain_chart():
    if st.session_state.chart is None:
        _show_error("Create a visualization first.")
        return
    if not _require_llm():
        return
    try:
        with st.spinner("Explaining chart (1 LLM call)…"):
            summary = get_result_summary(st.session_state.result_df)
            st.session_state.chart_explanation = explain_chart(
                st.session_state.chart_type,
                st.session_state.current_question,
                summary,
                st.session_state.profile,
            )
            _set_status("chart_explain", "done", "ok")
            st.session_state.last_error = None
    except LLMError as e:
        _show_error(str(e))


def _handle_executive_summary():
    if not _require_llm():
        return
    try:
        with st.spinner("Generating executive summary (1 LLM call)…"):
            st.session_state.exec_summary = generate_executive_summary(
                st.session_state.profile,
                st.session_state.qa_history,
            )
            _set_status("exec_summary", "generated", "ok")
            st.session_state.last_error = None
    except LLMError as e:
        _show_error(str(e))


def _save_to_history():
    if st.session_state.result_df is None:
        return
    entry = {
        "id": uuid.uuid4().hex,
        "question": st.session_state.current_question,
        "context": st.session_state.question_context,
        "code_info": st.session_state.generated_query,
        "result_df": st.session_state.result_df.copy(),
        "chart": st.session_state.chart,
        "chart_type": st.session_state.chart_type,
        "insight": st.session_state.insights,
        "exec_error": st.session_state.exec_error,
        "chart_explanation": st.session_state.chart_explanation,
    }
    st.session_state.qa_history.append(entry)
    if st.session_state.chart:
        try:
            st.session_state.chart_images.append(fig_to_bytes(st.session_state.chart))
        except Exception:
            pass


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ InsightPilot")
    st.caption("Universal data analytics — any schema")

    _render_status_bar()
    st.divider()

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "json", "txt", "parquet", "pq"],
        label_visibility="collapsed",
    )

    sample = st.selectbox("Sample data", ["— none —", "Weather observations", "Product inventory"])
    if st.button("Load sample", use_container_width=True):
        samples = {
            "Weather observations": "sample_data/weather_observations.csv",
            "Product inventory": "sample_data/product_inventory.csv",
        }
        if sample in samples:
            path = Path(__file__).parent / samples[sample]
            df, err = load_path(path)
            if err:
                _show_error(err)
            else:
                _load_dataset(df, path.name)
                st.success(f"Loaded {path.name}")
                st.rerun()

    if st.session_state.profile:
        st.divider()
        p = st.session_state.profile
        st.markdown(f"**{p.get('filename')}**")
        st.caption(f"{p.get('rows', 0):,} rows · {p.get('columns', 0)} columns")
        q = p.get("quality_score", 0)
        st.progress(min(q, 100) / 100, text=f"Quality {q}/100")

        if st.button("Clear session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()


# Handle upload
if uploaded is not None:
    fkey = f"{uploaded.name}_{uploaded.size}"
    if st.session_state._last_file != fkey:
        st.session_state._last_file = fkey
        with st.spinner(f"Loading {uploaded.name}…"):
            df, err = load_file(uploaded)
        if err:
            _show_error(err)
        else:
            _load_dataset(df, uploaded.name)
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("## Welcome to InsightPilot")
    st.markdown(
        "Upload **any** CSV, Excel, JSON, or Parquet file. "
        "The platform profiles your schema locally, then you control every AI step."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Profile locally** — no API calls on upload")
    c2.markdown("**2. Generate & run queries** — learn SQL/Pandas step by step")
    c3.markdown("**3. Insights on demand** — pay for AI only when you click")
    st.info("Set `LLM_PROVIDER=groq|gemini|openrouter` in `.env` to choose your AI provider.")
else:
    profile = st.session_state.profile
    st.markdown(f"## {profile.get('filename', 'Dataset')}")
    _render_kpis()

    tab_overview, tab_query, tab_results, tab_viz, tab_insights, tab_exec = st.tabs([
        "📋 Dataset Overview",
        "🔧 Query Builder",
        "📊 Results",
        "📈 Visualizations",
        "💡 Insights",
        "📝 Executive Summary",
    ])

    # ── Tab: Overview ─────────────────────────────────────────────
    with tab_overview:
        st.markdown(st.session_state.local_summary)
        if st.session_state.suggested_questions:
            st.markdown("**Suggested questions** (schema-driven, click to use)")
            cols = st.columns(2)
            for i, q in enumerate(st.session_state.suggested_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.current_question = q
                        st.session_state.last_error = None

        with st.expander("Column profiles", expanded=False):
            cp_df = pd.DataFrame(profile.get("column_profiles", []))
            if not cp_df.empty:
                show_cols = [c for c in ["name", "type", "dtype", "null_pct", "unique_count"] if c in cp_df.columns]
                st.dataframe(cp_df[show_cols], use_container_width=True, height=280)

        with st.expander("Raw data preview", expanded=False):
            st.dataframe(st.session_state.df.head(100), use_container_width=True, height=300)

    # ── Tab: Query Builder ────────────────────────────────────────
    with tab_query:
        st.session_state.current_question = st.text_area(
            "Your question",
            value=st.session_state.current_question,
            placeholder="e.g. Find records where city equals Pune | Which category has the highest total amount?",
            height=90,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⚡ Generate Query", use_container_width=True, type="primary"):
                _handle_generate_query()
        with c2:
            if st.button("▶ Run Query", use_container_width=True):
                _handle_run_query()
        with c3:
            if st.button("💾 Save to history", use_container_width=True):
                _save_to_history()
                st.success("Saved to analysis history.")

        if st.session_state.question_context:
            ctx = st.session_state.question_context
            st.markdown('<div class="panel"><div class="panel-title">Question understanding</div>', unsafe_allow_html=True)
            st.markdown(
                f"**Intent:** `{ctx.get('intent_type')}` · "
                f"**Method:** {ctx.get('analysis_method')} · "
                f"**Engine:** {ctx.get('execution_approach')}"
            )
            st.caption(ctx.get("reasoning", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.generated_query:
            qi = st.session_state.generated_query
            lang = "sql" if qi.get("code_type") == "sql" else "python"
            st.markdown('<div class="panel"><div class="panel-title">Generated query — inspect before running</div>', unsafe_allow_html=True)
            st.code(qi.get("code", ""), language=lang)
            if qi.get("explanation"):
                st.caption(f"📖 {qi['explanation']}")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.status:
            with st.expander("Pipeline status", expanded=False):
                for k, v in st.session_state.status.items():
                    st.write(f"**{k}** — {v['value']} ({v['at']})")

    # ── Tab: Results ──────────────────────────────────────────────
    with tab_results:
        if st.session_state.exec_error and st.session_state.result_df is None:
            st.warning(st.session_state.exec_error)
        elif st.session_state.result_df is not None:
            if st.session_state.result_df.empty:
                st.info("Query returned no rows. Try adjusting filters or column names.")
            else:
                st.success(f"{len(st.session_state.result_df):,} rows × {st.session_state.result_df.shape[1]} columns")
                st.dataframe(st.session_state.result_df, use_container_width=True, height=380)
                buf = io.StringIO()
                st.session_state.result_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇ Download CSV",
                    data=buf.getvalue(),
                    file_name="insightpilot_results.csv",
                    mime="text/csv",
                )
        else:
            st.info("Generate and run a query to see results here.")

    # ── Tab: Visualizations ───────────────────────────────────────
    with tab_viz:
        if st.session_state.result_df is not None and not st.session_state.result_df.empty:
            st.session_state.chart_type = st.selectbox(
                "Chart type",
                CHART_OPTIONS,
                index=CHART_OPTIONS.index(st.session_state.chart_type)
                if st.session_state.chart_type in CHART_OPTIONS else 0,
            )
            if st.button("Render chart", use_container_width=True):
                fig, resolved, chart_err = create_visualization(
                    st.session_state.result_df,
                    st.session_state.chart_type,
                    st.session_state.current_question,
                    profile,
                    st.session_state.question_context,
                )
                st.session_state.chart = fig
                st.session_state.chart_type = resolved
                st.session_state.chart_error = chart_err

            if st.session_state.chart_error:
                st.warning(st.session_state.chart_error)
            if st.session_state.chart:
                st.plotly_chart(st.session_state.chart, use_container_width=True)
                if st.button("💡 Explain Chart", use_container_width=True):
                    _handle_explain_chart()
                if st.session_state.chart_explanation:
                    st.info(st.session_state.chart_explanation)
        else:
            st.info("Run a query with results to create visualizations.")

    # ── Tab: Insights ─────────────────────────────────────────────
    with tab_insights:
        if st.button("💡 Generate Insights", type="primary", use_container_width=True):
            _handle_generate_insights()

        ins = st.session_state.insights
        if ins:
            conf = ins.get("confidence_level", "Medium")
            st.markdown(f"**Confidence:** {conf}")
            for f in ins.get("key_findings", []):
                st.markdown(f"- {f}")
            if ins.get("pattern"):
                st.markdown(f"**Pattern:** {ins['pattern']}")
            if ins.get("anomaly"):
                st.markdown(f"**Anomaly:** {ins['anomaly']}")
            if ins.get("recommendation"):
                st.success(f"💡 {ins['recommendation']}")
        else:
            st.info("Click **Generate Insights** after running a query. No automatic AI calls.")

        if st.session_state.qa_history:
            with st.expander("Analysis history"):
                for h in reversed(st.session_state.qa_history[-10:]):
                    st.markdown(f"**Q:** {h['question']}")
                    if h.get("result_df") is not None:
                        st.caption(f"{len(h['result_df'])} rows returned")

    # ── Tab: Executive Summary ────────────────────────────────────
    with tab_exec:
        if st.button("📝 Generate Executive Summary", type="primary", use_container_width=True):
            _handle_executive_summary()

        if st.session_state.exec_summary:
            st.markdown(st.session_state.exec_summary)
        else:
            st.info("Generate an executive summary on demand. Requires at least one saved analysis for best results.")

        if st.session_state.qa_history and st.button("📄 Export PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                summary = st.session_state.exec_summary or "Analysis session report."
                pdf = generate_pdf_report(
                    profile,
                    st.session_state.qa_history,
                    st.session_state.chart_images,
                    summary,
                )
            st.download_button(
                "⬇ Download PDF",
                data=pdf,
                file_name=f"InsightPilot_{profile.get('filename', 'report').rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
            )
