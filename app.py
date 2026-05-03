"""
============================================================
SmartCFO — Main Streamlit Application
Pakistani SME Financial Dashboard
Deployable on Hugging Face Spaces
============================================================

Entry point for the SmartCFO dashboard. Run with:
    streamlit run app.py

HuggingFace Spaces deployment:
    - Set GROQ_API_KEY and/or GEMINI_API_KEY in Space secrets
    - requirements.txt must be in the root directory
    - This file must be named app.py for HF Spaces auto-detection
"""

import sys
import os

# Add project root to path (important for HF Spaces)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from typing import Optional

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="SmartCFO — Pakistani SME Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/smartcfo",
        "Report a Bug": "https://github.com/your-repo/smartcfo/issues",
        "About": "SmartCFO — AI-powered financial dashboard for Pakistani SMEs",
    },
)

# ─── Imports (after sys.path setup) ───────────────────────────────────────────

from data.dummy_data import (
    generate_full_year_data,
    generate_monthly_receipts,
    COMPANY_NAME,
    EXPENSE_CATEGORIES,
    MONTHLY_REVENUE_BASE,
)
from utils.pl_engine import PLEngine, PLStatement
from utils.ocr_pipeline import process_receipt_text
from utils.visualizations import (
    plot_monthly_pl,
    plot_expense_pie,
    plot_margin_trends,
    plot_pl_waterfall,
    plot_ytd_cumulative,
    plot_expense_heatmap,
    compute_kpi_cards,
    fmt_pkr,
)
from utils.ai_integration import (
    SmartCFOAIRouter,
    mock_ai_extract,
    mock_cfo_insights,
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0F172A;
        --bg-card: #1E293B;
        --border: #334155;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --emerald: #10B981;
        --rose: #F43F5E;
        --amber: #F59E0B;
        --blue: #3B82F6;
        --purple: #8B5CF6;
        --cyan: #06B6D4;
    }

    /* App background */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0D1829;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }

    /* KPI Cards */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: var(--blue);
    }
    .kpi-label {
        color: var(--text-secondary);
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: var(--text-primary);
        font-size: 26px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 6px;
    }
    .kpi-delta-pos {
        color: var(--emerald);
        font-size: 13px;
        font-weight: 500;
    }
    .kpi-delta-neg {
        color: var(--rose);
        font-size: 13px;
        font-weight: 500;
    }
    .kpi-delta-neutral {
        color: var(--text-secondary);
        font-size: 13px;
    }

    /* Alert Cards */
    .alert-critical {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-left: 4px solid var(--rose);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid var(--amber);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-positive {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid var(--emerald);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-info {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 14px;
    }
    .alert-msg {
        color: var(--text-secondary);
        font-size: 13px;
        margin-top: 4px;
    }
    .alert-rec {
        color: var(--cyan);
        font-size: 12px;
        margin-top: 6px;
        font-style: italic;
    }

    /* Receipt viewer */
    .receipt-text {
        background: #0D1117;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #7DD3FC;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Section headers */
    .section-header {
        color: var(--text-primary);
        font-size: 18px;
        font-weight: 600;
        border-bottom: 1px solid var(--border);
        padding-bottom: 12px;
        margin: 24px 0 16px 0;
    }

    /* Tables */
    .stDataFrame {
        background: var(--bg-card);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1D4ED8 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        font-family: 'Inter', sans-serif;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Streamlit selectbox */
    .stSelectbox > div > div {
        background-color: var(--bg-card);
        border-color: var(--border);
        color: var(--text-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Logo area */
    .logo-area {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .logo-title {
        color: var(--text-primary);
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .logo-subtitle {
        color: var(--text-secondary);
        font-size: 12px;
    }

    /* Provider badge */
    .provider-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: var(--blue);
        font-weight: 500;
    }
</style>
"""

# ─── Session State Initialization ─────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "year_data" not in st.session_state:
        st.session_state.year_data = None
    if "statements" not in st.session_state:
        st.session_state.statements = None
    if "annual_summary" not in st.session_state:
        st.session_state.annual_summary = None
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = "Jan"
    if "ai_router" not in st.session_state:
        st.session_state.ai_router = None
    if "extracted_results" not in st.session_state:
        st.session_state.extracted_results = {}
    if "ai_insights_cache" not in st.session_state:
        st.session_state.ai_insights_cache = {}


# ─── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_year_data(year: int = 2024):
    """Cached data loader for the full year dataset."""
    return generate_full_year_data(year)


@st.cache_resource(show_spinner=False)
def get_pl_engine() -> PLEngine:
    """Cached P&L engine instance."""
    return PLEngine()


def get_ai_router() -> SmartCFOAIRouter:
    """Get AI router with API keys from environment/secrets."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    return SmartCFOAIRouter(
        groq_api_key=groq_key or None,
        gemini_api_key=gemini_key or None,
    )


# ─── UI Components ────────────────────────────────────────────────────────────

def render_kpi_cards(stmt: PLStatement, prev_stmt: Optional[PLStatement] = None):
    """Renders the KPI metric card row."""
    cards = compute_kpi_cards(stmt, prev_stmt)
    cols = st.columns(len(cards))

    for col, card in zip(cols, cards):
        with col:
            delta_html = ""
            if card.get("delta"):
                is_positive = card.get("positive_delta", True)
                delta_val = card["delta"]
                if "+" in str(delta_val) or (is_positive and not str(delta_val).startswith("-")):
                    delta_class = "kpi-delta-pos"
                    arrow = "↑"
                elif "-" in str(delta_val):
                    delta_class = "kpi-delta-neg"
                    arrow = "↓"
                else:
                    delta_class = "kpi-delta-neutral"
                    arrow = "→"
                delta_html = f'<div class="{delta_class}">{arrow} {delta_val} MoM</div>'

            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{card['icon']} {card['label']}</div>
                <div class="kpi-value">{card['value']}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)


def render_cfo_alerts(stmt: PLStatement):
    """Renders automated CFO alerts."""
    engine = get_pl_engine()
    alerts = engine.generate_cfo_alerts(stmt)

    for alert in alerts[:5]:  # Show top 5 alerts
        css_class = f"alert-{alert.severity}"
        st.markdown(f"""
        <div class="{css_class}">
            <div class="alert-title">{alert.icon} {alert.title}</div>
            <div class="alert-msg">{alert.message}</div>
            <div class="alert-rec">💡 {alert.recommendation}</div>
        </div>
        """, unsafe_allow_html=True)


def render_receipt_explorer(month: str, year_data: dict, ai_router: SmartCFOAIRouter):
    """Renders the receipt explorer with OCR extraction and AI analysis."""
    receipts = year_data["receipts"].get(month, [])

    if not receipts:
        st.warning("No receipts found for this month.")
        return

    # Receipt selector
    receipt_options = {
        f"{r.receipt_id} — {r.vendor} ({fmt_pkr(r.amount)})": r
        for r in receipts
    }
    selected_label = st.selectbox(
        "Select Receipt to Analyze",
        options=list(receipt_options.keys()),
        key=f"receipt_selector_{month}",
    )
    selected_receipt = receipt_options[selected_label]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">📄 Raw Invoice Text (OCR Simulation)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="receipt-text">{selected_receipt.raw_text}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="section-header">🔍 Extraction Results</div>',
                    unsafe_allow_html=True)

        cache_key = selected_receipt.receipt_id
        if cache_key not in st.session_state.extracted_results:
            if st.button("🤖 Extract with AI", key=f"extract_{cache_key}"):
                with st.spinner("Processing receipt..."):
                    result = mock_ai_extract(selected_receipt.raw_text)
                    # Try real AI if available
                    if ai_router.available_providers[0] not in ["regex_fallback"]:
                        ai_result, provider = ai_router.extract_receipt(selected_receipt.raw_text)
                        if ai_result.is_valid:
                            result = ai_result
                    st.session_state.extracted_results[cache_key] = result
                st.rerun()
        else:
            result = st.session_state.extracted_results[cache_key]

            # Ground truth vs extracted
            st.markdown("**Ground Truth vs. Extracted:**")
            comparison_data = {
                "Field": ["Vendor", "Amount (PKR)", "Invoice #", "Date", "Category", "Confidence"],
                "Ground Truth": [
                    selected_receipt.vendor,
                    f"{selected_receipt.amount:,.2f}",
                    selected_receipt.invoice_number,
                    selected_receipt.date,
                    selected_receipt.category,
                    "100%",
                ],
                "AI Extracted": [
                    result.vendor_name or "—",
                    f"{result.amount_total:,.2f}" if result.amount_total else "—",
                    result.invoice_number or "—",
                    result.invoice_date or "—",
                    result.expense_category,
                    f"{result.extraction_confidence:.0%}",
                ],
            }
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Model info
            st.markdown(
                f'<span class="provider-badge">🤖 {result.model_used}</span>',
                unsafe_allow_html=True,
            )

            if result.ai_notes:
                st.info(f"💬 AI Notes: {result.ai_notes}")

            if result.errors:
                for err in result.errors:
                    st.warning(f"⚠️ {err}")


def render_pl_table(stmt: PLStatement):
    """Renders the detailed P&L statement as a formatted table."""
    rows = []
    rows.append({"Item": "Revenue (Net Sales)", "Amount (PKR)": f"{stmt.revenue:,.0f}", "% of Revenue": "100.0%"})
    rows.append({"Item": "(-) Cost of Goods Sold", "Amount (PKR)": f"({stmt.cogs:,.0f})", "% of Revenue": f"({stmt.cogs_ratio:.1f}%)"})
    rows.append({"Item": "━━ GROSS PROFIT", "Amount (PKR)": f"{stmt.gross_profit:,.0f}", "% of Revenue": f"{stmt.gross_margin:.1f}%"})
    rows.append({"Item": "", "Amount (PKR)": "", "% of Revenue": ""})
    rows.append({"Item": "Operating Expenses:", "Amount (PKR)": "", "% of Revenue": ""})

    for cat, amount in sorted(stmt.opex_breakdown.items(), key=lambda x: -x[1]):
        label = EXPENSE_CATEGORIES.get(cat, {}).get("label", cat)
        ratio = stmt.expense_ratios.get(cat, 0)
        rows.append({
            "Item": f"  ├─ {label}",
            "Amount (PKR)": f"({amount:,.0f})",
            "% of Revenue": f"({ratio:.1f}%)",
        })

    rows.append({"Item": f"  Total OpEx", "Amount (PKR)": f"({stmt.opex_total:,.0f})", "% of Revenue": f"({stmt.opex_ratio:.1f}%)"})
    rows.append({"Item": "", "Amount (PKR)": "", "% of Revenue": ""})
    rows.append({"Item": "━━ EBIT (Operating Profit)", "Amount (PKR)": f"{stmt.ebit:,.0f}", "% of Revenue": f"{stmt.ebit_margin:.1f}%"})
    rows.append({"Item": "    + Estimated Depreciation", "Amount (PKR)": f"{stmt.estimated_depreciation:,.0f}", "% of Revenue": ""})
    rows.append({"Item": "━━ EBITDA", "Amount (PKR)": f"{stmt.ebitda:,.0f}", "% of Revenue": f"{stmt.ebitda_margin:.1f}%"})
    rows.append({"Item": "", "Amount (PKR)": "", "% of Revenue": ""})
    rows.append({"Item": f"(-) Corporate Tax Provision (29%)", "Amount (PKR)": f"({stmt.tax_provision:,.0f})", "% of Revenue": ""})
    rows.append({"Item": "━━ NET PROFIT (PAT)", "Amount (PKR)": f"{stmt.net_profit:,.0f}", "% of Revenue": f"{stmt.net_margin:.1f}%"})

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


# ─── Main Application ─────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="logo-area">
            <div class="logo-title">💼 SmartCFO</div>
            <div class="logo-subtitle">Pakistani SME Financial Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Company info
        st.markdown(f"**Company:** {COMPANY_NAME}")
        st.markdown("**Fiscal Year:** 2024")
        st.markdown("**Currency:** PKR 🇵🇰")
        st.markdown("**Tax Rate:** 29% (FBR Corp. Tax)")

        st.divider()

        # Month selector
        months = list(MONTHLY_REVENUE_BASE.keys())
        selected_month = st.selectbox(
            "📅 Select Month",
            months,
            index=0,
            key="month_selector",
        )
        st.session_state.selected_month = selected_month

        st.divider()

        # API Key configuration
        st.markdown("**🤖 AI Configuration**")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get free key at console.groq.com",
            placeholder="gsk_...",
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Get key at aistudio.google.com",
            placeholder="AIza...",
        )

        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

        st.divider()
        st.caption("Built with Streamlit + Plotly")
        st.caption("Deploy on 🤗 Hugging Face Spaces")

    # ── Data Loading ───────────────────────────────────────────────────────────
    with st.spinner("📊 Loading financial data..."):
        year_data = load_year_data(2024)
        engine = get_pl_engine()
        statements = engine.process_year(year_data["financials"])
        annual_summary = engine.compute_annual_summary(statements, 2024)

    ai_router = get_ai_router()

    # Get current month statement
    month_idx = list(MONTHLY_REVENUE_BASE.keys()).index(selected_month)
    current_stmt = statements[month_idx]
    prev_stmt = statements[month_idx - 1] if month_idx > 0 else None

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding: 16px 0 8px 0;">
        <h1 style="color: #F1F5F9; font-size: 28px; font-weight: 800; margin: 0; letter-spacing: -0.03em;">
            💼 SmartCFO Dashboard
        </h1>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 14px;">
            {COMPANY_NAME} &nbsp;|&nbsp; {selected_month} 2024 &nbsp;|&nbsp;
            <span class="provider-badge">AI: {', '.join(ai_router.available_providers)}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Tab Layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 P&L Overview",
        "💰 Expense Analysis",
        "📈 Trends & Margins",
        "📄 Receipt Explorer",
        "🤖 CFO Insights",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: P&L OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Key Performance Indicators")
        render_kpi_cards(current_stmt, prev_stmt)

        st.divider()

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### Monthly Revenue vs Expenses vs Profit")
            fig_pl = plot_monthly_pl(statements)
            st.plotly_chart(fig_pl, use_container_width=True, key="monthly_pl")

        with col2:
            st.markdown(f"### P&L Waterfall — {selected_month}")
            fig_waterfall = plot_pl_waterfall(current_stmt)
            st.plotly_chart(fig_waterfall, use_container_width=True, key="waterfall")

        st.divider()
        st.markdown(f"### Detailed P&L Statement — {selected_month} 2024")
        render_pl_table(current_stmt)

        # Annual summary
        with st.expander("📋 Annual Summary 2024"):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Revenue", fmt_pkr(annual_summary.total_revenue, compact=True))
            col_b.metric("Total Net Profit", fmt_pkr(annual_summary.total_net_profit, compact=True))
            col_c.metric("Avg Net Margin", f"{annual_summary.avg_net_margin:.1f}%")
            col_d.metric("Best Month", annual_summary.best_month)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: EXPENSE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"### Expense Breakdown — {selected_month}")
            # Combine COGS and opex for pie chart
            full_breakdown = {**current_stmt.opex_breakdown, "cogs": current_stmt.cogs}
            fig_pie = plot_expense_pie(full_breakdown, period=f"{selected_month} 2024")
            st.plotly_chart(fig_pie, use_container_width=True, key="expense_pie")

        with col2:
            st.markdown("### CFO Alerts & Recommendations")
            render_cfo_alerts(current_stmt)

        st.divider()
        st.markdown("### Expense Ratio Heatmap (All Months)")
        st.caption("Color intensity = % of revenue consumed. Darker = higher spend ratio. Hover for details.")
        fig_heatmap = plot_expense_heatmap(statements)
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")

        # Expense table
        st.markdown(f"### Expense Detail Table — {selected_month}")
        expense_table_data = []
        full_breakdown_with_cogs = {**current_stmt.opex_breakdown, "cogs": current_stmt.cogs}
        total_exp = sum(full_breakdown_with_cogs.values())
        for cat, amount in sorted(full_breakdown_with_cogs.items(), key=lambda x: -x[1]):
            label = EXPENSE_CATEGORIES.get(cat, {}).get("label", cat)
            ratio = (amount / current_stmt.revenue * 100)
            expense_table_data.append({
                "Category": label,
                "Amount (PKR)": f"{amount:,.0f}",
                "% of Revenue": f"{ratio:.1f}%",
                "% of Total Expenses": f"{(amount/total_exp*100):.1f}%",
            })
        st.dataframe(
            pd.DataFrame(expense_table_data),
            hide_index=True,
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: TRENDS & MARGINS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Margin Trends — 2024")
        fig_margins = plot_margin_trends(statements)
        st.plotly_chart(fig_margins, use_container_width=True, key="margins")

        st.divider()

        # Forecast
        forecasts = engine.forecast_revenue(statements, periods_ahead=3)
        st.markdown("### Year-to-Date Revenue & Profit (with 3-Month Forecast)")
        fig_ytd = plot_ytd_cumulative(statements, forecasts=forecasts)
        st.plotly_chart(fig_ytd, use_container_width=True, key="ytd")

        # MoM comparison table
        st.divider()
        st.markdown("### Month-over-Month Variance")
        mom_data = []
        for i, stmt in enumerate(statements):
            prev = statements[i-1] if i > 0 else None
            revenue_growth = stmt.revenue_growth_mom if i > 0 else 0
            mom_data.append({
                "Month": stmt.period,
                "Revenue (PKR)": f"{stmt.revenue:,.0f}",
                "Net Profit (PKR)": f"{stmt.net_profit:,.0f}",
                "Gross Margin": f"{stmt.gross_margin:.1f}%",
                "Net Margin": f"{stmt.net_margin:.1f}%",
                "MoM Revenue Δ": f"{revenue_growth:+.1f}%" if i > 0 else "—",
            })
        st.dataframe(pd.DataFrame(mom_data), hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: RECEIPT EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"### Receipt Explorer — {selected_month} 2024")
        st.caption(
            "Explore synthetic Pakistani SME receipts with simulated OCR extraction and AI analysis. "
            "Configure API keys in the sidebar for AI-powered extraction."
        )
        render_receipt_explorer(selected_month, year_data, ai_router)

        st.divider()

        # Batch results if available
        if st.session_state.extracted_results:
            st.markdown("### Extraction History")
            history_data = []
            for key, result in st.session_state.extracted_results.items():
                history_data.append({
                    "Receipt ID": key,
                    "Vendor": result.vendor_name or "—",
                    "Amount (PKR)": f"{result.amount_total:,.0f}" if result.amount_total else "—",
                    "Category": result.expense_category,
                    "Confidence": f"{result.extraction_confidence:.0%}",
                    "Model": result.model_used,
                })
            st.dataframe(pd.DataFrame(history_data), hide_index=True, use_container_width=True)

            if st.button("🗑️ Clear Extraction History"):
                st.session_state.extracted_results = {}
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: CFO INSIGHTS (AI-Powered)
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown(f"### AI-Powered CFO Insights — {selected_month} 2024")

        provider_info = ", ".join(ai_router.available_providers)
        if "regex_fallback" in provider_info and "groq" not in provider_info:
            st.info(
                "💡 **Demo Mode**: Running with rule-based insights. "
                "Add your **GROQ_API_KEY** or **GEMINI_API_KEY** in the sidebar "
                "for AI-powered analysis using LLaMA 3.1 or Gemini."
            )

        # Generate insights
        insight_cache_key = f"{selected_month}_insights"
        if insight_cache_key not in st.session_state.ai_insights_cache:
            with st.spinner("🧠 Generating CFO insights..."):
                pl_data = current_stmt.to_dict()
                pl_data["company_name"] = COMPANY_NAME

                if ai_router.available_providers[0] not in ["regex_fallback"]:
                    insights = ai_router.generate_cfo_insights(pl_data)
                else:
                    insights = mock_cfo_insights(pl_data)

                st.session_state.ai_insights_cache[insight_cache_key] = insights

        insights = st.session_state.ai_insights_cache[insight_cache_key]

        # Executive Summary
        st.markdown("#### 📋 Executive Summary")
        st.markdown(f"""
        <div class="alert-info">
            <div class="alert-msg" style="font-size: 15px; line-height: 1.7;">
                {insights.summary}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔴 Key Risk Factors")
            for i, risk in enumerate(insights.key_risks, 1):
                st.markdown(f"""
                <div class="alert-warning">
                    <div class="alert-title">Risk {i}</div>
                    <div class="alert-msg">{risk}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 💡 Cost Optimization Tips")
            for i, tip in enumerate(insights.cost_optimization_tips, 1):
                st.markdown(f"""
                <div class="alert-positive">
                    <div class="alert-title">Tip {i}</div>
                    <div class="alert-msg">{tip}</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### 📈 Revenue Analysis")
            st.markdown(f"""
            <div class="alert-info">
                <div class="alert-msg" style="line-height: 1.7;">{insights.revenue_observations}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 💵 Cashflow Outlook")
            st.markdown(f"""
            <div class="alert-info">
                <div class="alert-msg" style="line-height: 1.7;">{insights.cashflow_outlook}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🏛️ FBR / Tax Notes")
            st.markdown(f"""
            <div class="alert-critical" style="border-left-color: #F59E0B;">
                <div class="alert-msg" style="line-height: 1.7;">{insights.tax_notes}</div>
            </div>
            """, unsafe_allow_html=True)

        # Refresh button
        if st.button("🔄 Regenerate Insights"):
            if insight_cache_key in st.session_state.ai_insights_cache:
                del st.session_state.ai_insights_cache[insight_cache_key]
            st.rerun()

        st.divider()
        st.caption(f"Generated by: {insights.model_used} | SmartCFO v1.0")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
