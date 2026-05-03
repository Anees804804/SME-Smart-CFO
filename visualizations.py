"""
============================================================
SmartCFO — Data Visualization Module
Plotly-powered Financial Charts for Pakistani SME Dashboard
============================================================

This module provides production-grade Plotly figures for:
1. P&L Waterfall Chart          → Visual profit decomposition
2. Monthly Revenue vs Net Profit → Dual-axis bar + line chart
3. Expense Breakdown Pie Chart  → Category-wise spend distribution
4. Gross/Net Margin Trend        → Line chart over 12 months
5. YTD Cumulative Revenue Chart  → Area chart with forecast overlay
6. Expense Ratio Heatmap         → Category × Month heatmap

Design Principles:
- Dark theme optimized for financial dashboards (#0F172A background)
- Pakistan Rupee (PKR) formatting on all axes
- Interactive tooltips with CFO-grade context
- Mobile-responsive layout configs
- Color scheme: Emerald for positive, Rose for negative metrics
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import statistics

from data.dummy_data import EXPENSE_CATEGORIES
from utils.pl_engine import PLStatement, AnnualSummary


# ─── Design Tokens ────────────────────────────────────────────────────────────

DARK_BG = "#0F172A"
CARD_BG = "#1E293B"
BORDER = "#334155"
TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
ACCENT_EMERALD = "#10B981"
ACCENT_ROSE = "#F43F5E"
ACCENT_AMBER = "#F59E0B"
ACCENT_BLUE = "#3B82F6"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_CYAN = "#06B6D4"

# Category colors (from EXPENSE_CATEGORIES, fallback palette)
CAT_COLORS = {cat: data["color"] for cat, data in EXPENSE_CATEGORIES.items()}
CAT_COLORS["uncategorized"] = "#64748B"

# Pakistani number formatting
def fmt_pkr(amount: float, compact: bool = False) -> str:
    """Formats PKR amounts in South Asian format (Lakhs/Crores)."""
    if compact:
        if abs(amount) >= 10_000_000:
            return f"PKR {amount/10_000_000:.1f}Cr"
        elif abs(amount) >= 100_000:
            return f"PKR {amount/100_000:.1f}L"
        elif abs(amount) >= 1000:
            return f"PKR {amount/1000:.0f}K"
        return f"PKR {amount:,.0f}"
    return f"PKR {amount:,.0f}"

def pct_fmt(value: float) -> str:
    return f"{value:+.1f}%" if value != 0 else "0.0%"


# ─── Base Layout ──────────────────────────────────────────────────────────────

def _base_layout(title: str, height: int = 420) -> dict:
    """Returns the base dark-theme layout config."""
    return dict(
        title=dict(
            text=title,
            font=dict(color=TEXT_PRIMARY, size=16, family="Inter, sans-serif"),
            x=0.02,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_SECONDARY, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=True,
        legend=dict(
            bgcolor=DARK_BG,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT_SECONDARY, size=11),
        ),
        xaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY),
            zerolinecolor=BORDER,
        ),
        yaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY),
            zerolinecolor=BORDER,
        ),
        hoverlabel=dict(
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
    )


# ─── 1. Monthly P&L Bar Chart ─────────────────────────────────────────────────

def plot_monthly_pl(statements: List[PLStatement]) -> go.Figure:
    """
    Grouped bar chart showing Revenue, Total Expenses, and Net Profit
    for each month. Net Profit bars are color-coded (green = profit, red = loss).

    CFO Use Case:
    At a glance, identifies which months had profit compression,
    and whether cost growth is outpacing revenue growth.
    """
    months = [s.period for s in statements]
    revenues = [s.revenue for s in statements]
    total_expenses = [s.total_expenses for s in statements]
    net_profits = [s.net_profit for s in statements]
    gross_profits = [s.gross_profit for s in statements]

    # Color net profit bars conditionally
    net_colors = [ACCENT_EMERALD if p >= 0 else ACCENT_ROSE for p in net_profits]

    fig = go.Figure()

    # Revenue bars
    fig.add_trace(go.Bar(
        name="Revenue",
        x=months,
        y=revenues,
        marker_color=ACCENT_BLUE,
        marker_opacity=0.85,
        customdata=[[fmt_pkr(r), fmt_pkr(e), fmt_pkr(n)]
                    for r, e, n in zip(revenues, total_expenses, net_profits)],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Revenue: %{customdata[0]}<br>"
            "Expenses: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    ))

    # Total Expenses bars
    fig.add_trace(go.Bar(
        name="Total Expenses",
        x=months,
        y=total_expenses,
        marker_color=ACCENT_ROSE,
        marker_opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Expenses: PKR %{y:,.0f}<extra></extra>",
    ))

    # Gross Profit line
    fig.add_trace(go.Scatter(
        name="Gross Profit",
        x=months,
        y=gross_profits,
        mode="lines+markers",
        line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
        marker=dict(size=5, color=ACCENT_AMBER),
        hovertemplate="<b>%{x}</b><br>Gross Profit: PKR %{y:,.0f}<extra></extra>",
    ))

    # Net Profit line
    fig.add_trace(go.Scatter(
        name="Net Profit",
        x=months,
        y=net_profits,
        mode="lines+markers",
        line=dict(color=ACCENT_EMERALD, width=2.5),
        marker=dict(size=7, color=net_colors, line=dict(color=DARK_BG, width=1)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Net Profit: PKR %{y:,.0f}<extra></extra>"
        ),
    ))

    layout = _base_layout("Monthly Revenue vs Expenses vs Net Profit (PKR)", height=440)
    layout["barmode"] = "group"
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["yaxis"]["tickprefix"] = "PKR "
    fig.update_layout(**layout)

    return fig


# ─── 2. Expense Breakdown Pie Chart ──────────────────────────────────────────

def plot_expense_pie(
    expense_breakdown: Dict[str, float],
    period: str = "Selected Period",
) -> go.Figure:
    """
    Donut chart showing proportional spend across expense categories.
    
    CFO Use Case:
    Quickly identifies which categories consume the most cash —
    helps focus cost optimization efforts.
    """
    # Filter zero-value categories
    filtered = {k: v for k, v in expense_breakdown.items() if v > 0}
    sorted_items = sorted(filtered.items(), key=lambda x: -x[1])

    labels = [EXPENSE_CATEGORIES.get(k, {}).get("label", k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [CAT_COLORS.get(k, "#64748B") for k, _ in sorted_items]
    total = sum(values)

    customdata = [
        [fmt_pkr(v), f"{v/total*100:.1f}%"] for v in values
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(
            colors=colors,
            line=dict(color=DARK_BG, width=2),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Amount: %{customdata[0]}<br>"
            "Share: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        textinfo="label+percent",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        pull=[0.03 if i == 0 else 0 for i in range(len(labels))],
    ))

    # Center annotation
    fig.add_annotation(
        text=f"<b>Total</b><br>{fmt_pkr(total, compact=True)}",
        x=0.5, y=0.5,
        font=dict(size=13, color=TEXT_PRIMARY),
        showarrow=False,
        xanchor="center",
    )

    layout = _base_layout(f"Expense Breakdown — {period}", height=440)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout["showlegend"] = True
    layout["legend"]["orientation"] = "v"
    fig.update_layout(**layout)

    return fig


# ─── 3. Margin Trend Line Chart ───────────────────────────────────────────────

def plot_margin_trends(statements: List[PLStatement]) -> go.Figure:
    """
    Multi-line chart showing Gross, EBIT, and Net margin trends.
    Includes a reference line at 0% (break-even) and target lines.
    
    CFO Use Case:
    Tracks margin sustainability over time — crucial for evaluating
    whether profitability is structural or situational.
    """
    months = [s.period for s in statements]
    gross_margins = [s.gross_margin for s in statements]
    ebit_margins = [s.ebit_margin for s in statements]
    net_margins = [s.net_margin for s in statements]
    ebitda_margins = [s.ebitda_margin for s in statements]

    fig = go.Figure()

    # Gross Margin
    fig.add_trace(go.Scatter(
        name="Gross Margin",
        x=months, y=gross_margins,
        mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x}</b><br>Gross Margin: %{y:.1f}%<extra></extra>",
    ))

    # EBITDA Margin
    fig.add_trace(go.Scatter(
        name="EBITDA Margin",
        x=months, y=ebitda_margins,
        mode="lines+markers",
        line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>EBITDA Margin: %{y:.1f}%<extra></extra>",
    ))

    # EBIT Margin
    fig.add_trace(go.Scatter(
        name="EBIT Margin",
        x=months, y=ebit_margins,
        mode="lines+markers",
        line=dict(color=ACCENT_PURPLE, width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>EBIT Margin: %{y:.1f}%<extra></extra>",
    ))

    # Net Margin
    fig.add_trace(go.Scatter(
        name="Net Margin (PAT)",
        x=months, y=net_margins,
        mode="lines+markers",
        line=dict(color=ACCENT_EMERALD, width=3),
        marker=dict(
            size=8,
            color=[ACCENT_EMERALD if m >= 0 else ACCENT_ROSE for m in net_margins],
            line=dict(color=DARK_BG, width=1.5),
        ),
        hovertemplate="<b>%{x}</b><br>Net Margin: %{y:.1f}%<extra></extra>",
    ))

    # Reference Lines
    fig.add_hline(y=0, line_color=ACCENT_ROSE, line_width=1.5,
                  annotation_text="Break-even", annotation_font_color=ACCENT_ROSE)
    fig.add_hline(y=10, line_color=ACCENT_EMERALD, line_width=1,
                  line_dash="dash", annotation_text="10% Net Target",
                  annotation_font_color=ACCENT_EMERALD)

    layout = _base_layout("Margin Trends — Gross / EBITDA / EBIT / Net (%)", height=420)
    layout["yaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["title"] = "Margin (%)"
    fig.update_layout(**layout)

    return fig


# ─── 4. P&L Waterfall Chart ───────────────────────────────────────────────────

def plot_pl_waterfall(stmt: PLStatement) -> go.Figure:
    """
    Waterfall chart showing the full P&L decomposition for a period.
    
    Walks from Revenue → COGS deduction → Gross Profit →
    OpEx deduction (by category) → EBIT → Tax → Net Profit.
    
    CFO Use Case:
    The single most powerful chart for a CFO — shows exactly
    where revenue is being eroded at each layer.
    """
    # Build waterfall steps
    measures = []
    x_labels = []
    y_values = []
    texts = []
    colors_list = []

    # Revenue (absolute)
    measures.append("absolute")
    x_labels.append("Revenue")
    y_values.append(stmt.revenue)
    texts.append(fmt_pkr(stmt.revenue, compact=True))
    colors_list.append(ACCENT_BLUE)

    # COGS (relative negative)
    measures.append("relative")
    x_labels.append("(-) COGS")
    y_values.append(-stmt.cogs)
    texts.append(f"-{fmt_pkr(stmt.cogs, compact=True)}")
    colors_list.append(ACCENT_ROSE)

    # Gross Profit (subtotal)
    measures.append("total")
    x_labels.append("Gross Profit")
    y_values.append(stmt.gross_profit)
    texts.append(f"{fmt_pkr(stmt.gross_profit, compact=True)} ({stmt.gross_margin:.1f}%)")
    colors_list.append(ACCENT_AMBER)

    # OpEx categories (relative negative)
    opex_sorted = sorted(stmt.opex_breakdown.items(), key=lambda x: -x[1])
    for cat, amount in opex_sorted:
        label = EXPENSE_CATEGORIES.get(cat, {}).get("label", cat)
        measures.append("relative")
        x_labels.append(f"(-) {label}")
        y_values.append(-amount)
        texts.append(f"-{fmt_pkr(amount, compact=True)}")
        colors_list.append(CAT_COLORS.get(cat, "#64748B"))

    # EBIT (total)
    measures.append("total")
    x_labels.append("EBIT")
    y_values.append(stmt.ebit)
    texts.append(f"{fmt_pkr(stmt.ebit, compact=True)} ({stmt.ebit_margin:.1f}%)")
    colors_list.append(ACCENT_CYAN)

    # Tax (relative negative)
    measures.append("relative")
    x_labels.append("(-) Corp. Tax")
    y_values.append(-stmt.tax_provision)
    texts.append(f"-{fmt_pkr(stmt.tax_provision, compact=True)}")
    colors_list.append("#475569")

    # Net Profit (total)
    measures.append("total")
    x_labels.append("Net Profit")
    y_values.append(stmt.net_profit)
    net_color = ACCENT_EMERALD if stmt.net_profit >= 0 else ACCENT_ROSE
    texts.append(f"{fmt_pkr(stmt.net_profit, compact=True)} ({stmt.net_margin:.1f}%)")
    colors_list.append(net_color)

    fig = go.Figure(go.Waterfall(
        name="P&L Waterfall",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=texts,
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=10),
        connector=dict(
            line=dict(color=BORDER, width=1.5, dash="dot"),
        ),
        increasing=dict(marker=dict(color=ACCENT_EMERALD, line=dict(color=DARK_BG, width=1))),
        decreasing=dict(marker=dict(color=ACCENT_ROSE, line=dict(color=DARK_BG, width=1))),
        totals=dict(marker=dict(color=ACCENT_AMBER, line=dict(color=DARK_BG, width=1))),
        hovertemplate="<b>%{x}</b><br>PKR %{y:,.0f}<extra></extra>",
    ))

    layout = _base_layout(f"P&L Waterfall — {stmt.period}", height=520)
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["yaxis"]["tickprefix"] = "PKR "
    layout["showlegend"] = False
    layout["margin"]["b"] = 120
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-30)

    return fig


# ─── 5. YTD Cumulative Revenue & Profit ──────────────────────────────────────

def plot_ytd_cumulative(
    statements: List[PLStatement],
    forecasts: Optional[List[Dict]] = None,
) -> go.Figure:
    """
    Area chart of YTD cumulative Revenue and Net Profit.
    Optionally overlays a 3-month forecast.
    
    CFO Use Case:
    Tracks whether the business is on pace to meet annual targets.
    The gap between revenue and profit areas visually represents cost.
    """
    months = [s.period for s in statements]
    rev_ytd = []
    profit_ytd = []
    running_rev = 0
    running_profit = 0

    for s in statements:
        running_rev += s.revenue
        running_profit += s.net_profit
        rev_ytd.append(running_rev)
        profit_ytd.append(running_profit)

    fig = go.Figure()

    # Revenue area
    fig.add_trace(go.Scatter(
        name="YTD Revenue",
        x=months, y=rev_ytd,
        mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.12)",
        hovertemplate="<b>%{x}</b><br>YTD Revenue: PKR %{y:,.0f}<extra></extra>",
    ))

    # Net Profit area
    net_fill_color = "rgba(16,185,129,0.18)" if profit_ytd[-1] >= 0 else "rgba(244,63,94,0.18)"
    fig.add_trace(go.Scatter(
        name="YTD Net Profit",
        x=months, y=profit_ytd,
        mode="lines+markers",
        line=dict(color=ACCENT_EMERALD, width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor=net_fill_color,
        hovertemplate="<b>%{x}</b><br>YTD Net Profit: PKR %{y:,.0f}<extra></extra>",
    ))

    # Forecast overlay
    if forecasts:
        forecast_months = [f["period"] for f in forecasts]
        forecast_revenues = [f["forecasted_revenue"] for f in forecasts]
        # Extend YTD
        last_rev = rev_ytd[-1] if rev_ytd else 0
        extended_rev = [last_rev + sum(forecast_revenues[:i+1]) for i in range(len(forecasts))]

        all_x = [months[-1]] + forecast_months
        all_y = [rev_ytd[-1]] + extended_rev

        fig.add_trace(go.Scatter(
            name="Revenue Forecast",
            x=all_x, y=all_y,
            mode="lines+markers",
            line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="<b>%{x}</b> (Forecast)<br>PKR %{y:,.0f}<extra></extra>",
        ))

    layout = _base_layout("Year-to-Date Cumulative Revenue & Net Profit", height=400)
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["yaxis"]["tickprefix"] = "PKR "
    fig.update_layout(**layout)

    return fig


# ─── 6. Expense Ratio Heatmap ─────────────────────────────────────────────────

def plot_expense_heatmap(statements: List[PLStatement]) -> go.Figure:
    """
    Heatmap of expense ratios (category × month).
    Color intensity = % of revenue consumed by that category.
    
    CFO Use Case:
    The most powerful cost analysis view — immediately reveals
    which cost lines are creeping up relative to revenue.
    Any darkening column is a cost control red flag.
    """
    months = [s.period for s in statements]
    all_cats = list(EXPENSE_CATEGORIES.keys())
    cat_labels = [EXPENSE_CATEGORIES[c]["label"] for c in all_cats]

    # Build z-matrix: [categories × months]
    z_matrix = []
    for cat in all_cats:
        row = []
        for stmt in statements:
            ratio = stmt.expense_ratios.get(cat, 0)
            row.append(round(ratio, 2))
        z_matrix.append(row)

    # Hover text
    hover_text = []
    for cat, row in zip(all_cats, z_matrix):
        hover_row = []
        for month, ratio in zip(months, row):
            # Find absolute amount
            stmt = next((s for s in statements if s.period == month), None)
            if stmt:
                abs_amt = stmt.opex_breakdown.get(cat, stmt.cogs if cat == "cogs" else 0)
                hover_row.append(f"{EXPENSE_CATEGORIES[cat]['label']}<br>{month}<br>Ratio: {ratio:.1f}%<br>Amount: {fmt_pkr(abs_amt)}")
            else:
                hover_row.append(f"{ratio:.1f}%")
        hover_text.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=months,
        y=cat_labels,
        colorscale=[
            [0.0, CARD_BG],
            [0.2, "#1E3A5F"],
            [0.5, "#1D4ED8"],
            [0.75, "#7C3AED"],
            [1.0, ACCENT_ROSE],
        ],
        hoverinfo="text",
        text=hover_text,
        showscale=True,
        colorbar=dict(
            title="% of Revenue",
            titlefont=dict(color=TEXT_SECONDARY),
            tickfont=dict(color=TEXT_SECONDARY),
            bgcolor=DARK_BG,
            bordercolor=BORDER,
        ),
        xgap=2,
        ygap=2,
    ))

    layout = _base_layout("Expense Ratio Heatmap — % of Revenue by Category & Month", height=480)
    layout["xaxis"]["side"] = "bottom"
    fig.update_layout(**layout)

    return fig


# ─── 7. KPI Metric Cards (returns dict for Streamlit) ─────────────────────────

def compute_kpi_cards(
    stmt: PLStatement,
    prev_stmt: Optional[PLStatement] = None,
) -> List[Dict]:
    """
    Computes KPI card data for the dashboard header.
    Returns list of dicts with label, value, delta, color.
    """
    def delta(curr, prev, is_pct=False):
        if prev is None or prev == 0:
            return None
        change = curr - prev
        if is_pct:
            return f"{change:+.1f}pp"
        pct_chg = (change / abs(prev)) * 100
        return f"{pct_chg:+.1f}%"

    prev_revenue = prev_stmt.revenue if prev_stmt else None
    prev_net = prev_stmt.net_profit if prev_stmt else None
    prev_gross = prev_stmt.gross_margin if prev_stmt else None
    prev_net_margin = prev_stmt.net_margin if prev_stmt else None

    cards = [
        {
            "label": "Revenue",
            "value": fmt_pkr(stmt.revenue, compact=True),
            "value_full": fmt_pkr(stmt.revenue),
            "delta": delta(stmt.revenue, prev_revenue),
            "color": ACCENT_BLUE,
            "icon": "💰",
            "positive_delta": True,
        },
        {
            "label": "Gross Profit",
            "value": fmt_pkr(stmt.gross_profit, compact=True),
            "value_full": fmt_pkr(stmt.gross_profit),
            "delta": f"{stmt.gross_margin:.1f}%",
            "delta_secondary": delta(stmt.gross_margin, prev_gross, is_pct=True),
            "color": ACCENT_AMBER,
            "icon": "📊",
            "positive_delta": True,
        },
        {
            "label": "Net Profit",
            "value": fmt_pkr(stmt.net_profit, compact=True),
            "value_full": fmt_pkr(stmt.net_profit),
            "delta": delta(stmt.net_profit, prev_net),
            "color": ACCENT_EMERALD if stmt.net_profit >= 0 else ACCENT_ROSE,
            "icon": "✅" if stmt.net_profit >= 0 else "⚠️",
            "positive_delta": stmt.net_profit >= 0,
        },
        {
            "label": "Net Margin",
            "value": f"{stmt.net_margin:.1f}%",
            "value_full": f"{stmt.net_margin:.2f}%",
            "delta": delta(stmt.net_margin, prev_net_margin, is_pct=True),
            "color": ACCENT_PURPLE,
            "icon": "📈",
            "positive_delta": stmt.net_margin >= 0,
        },
        {
            "label": "Total Expenses",
            "value": fmt_pkr(stmt.total_expenses, compact=True),
            "value_full": fmt_pkr(stmt.total_expenses),
            "delta": delta(stmt.total_expenses, prev_stmt.total_expenses if prev_stmt else None),
            "color": ACCENT_ROSE,
            "icon": "🔻",
            "positive_delta": False,  # Lower expenses = better
        },
        {
            "label": "EBITDA",
            "value": fmt_pkr(stmt.ebitda, compact=True),
            "value_full": fmt_pkr(stmt.ebitda),
            "delta": f"{stmt.ebitda_margin:.1f}%",
            "color": ACCENT_CYAN,
            "icon": "🏦",
            "positive_delta": stmt.ebitda >= 0,
        },
    ]
    return cards


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.dummy_data import generate_full_year_data
    from utils.pl_engine import PLEngine

    print("🎨 Testing SmartCFO Visualization Module...")
    data = generate_full_year_data(2024)
    engine = PLEngine()
    statements = engine.process_year(data["financials"])

    # Test each chart
    fig1 = plot_monthly_pl(statements)
    print(f"✅ Monthly P&L chart: {len(fig1.data)} traces")

    fig2 = plot_expense_pie(statements[0].opex_breakdown, period="January 2024")
    print(f"✅ Expense pie chart: {len(fig2.data)} traces")

    fig3 = plot_margin_trends(statements)
    print(f"✅ Margin trends chart: {len(fig3.data)} traces")

    fig4 = plot_pl_waterfall(statements[0])
    print(f"✅ P&L waterfall chart: {len(fig4.data)} traces")

    fig5 = plot_ytd_cumulative(statements)
    print(f"✅ YTD cumulative chart: {len(fig5.data)} traces")

    fig6 = plot_expense_heatmap(statements)
    print(f"✅ Expense heatmap: {len(fig6.data)} traces")

    kpis = compute_kpi_cards(statements[0], statements[1] if len(statements) > 1 else None)
    print(f"✅ KPI cards: {len(kpis)} metrics")

    print("\n✨ All visualization tests passed!")
