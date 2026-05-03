"""
============================================================
SmartCFO — P&L Engine & CFO Formula Library
============================================================

This module implements the complete financial intelligence layer.

Accounting Standards Applied:
- Based on IAS 1 (Presentation of Financial Statements)
- Adapted for Pakistani SME context (SECP + FBR regulations)
- Cash-basis accounting with accrual adjustments

P&L Structure (Pakistani SME):
┌─────────────────────────────────────┐
│  Revenue (Net Sales)                │
│  - Cost of Goods Sold (COGS)        │
│ ─────────────────────────────────── │
│  = GROSS PROFIT                     │
│  - Operating Expenses               │
│    ├─ Salaries & Benefits           │
│    ├─ Rent                          │
│    ├─ Utilities (K-Electric, SSGC)  │
│    ├─ Internet & Telecom            │
│    ├─ Marketing                     │
│    ├─ Software Subscriptions        │
│    ├─ Office Supplies               │
│    ├─ Transport & Logistics         │
│    └─ Professional Fees             │
│ ─────────────────────────────────── │
│  = OPERATING PROFIT (EBIT)          │
│  - Tax Provision (29% Corp. Tax)    │
│ ─────────────────────────────────── │
│  = NET PROFIT (PAT)                 │
└─────────────────────────────────────┘

Key Ratios Calculated:
- Gross Margin, Net Margin, Operating Margin
- Expense Ratio per category
- Month-over-Month (MoM) Growth
- Year-to-Date (YTD) Cumulative totals
- Burn Rate (monthly cash consumption)
- Runway (months of cash remaining)
- EBITDA (adds back depreciation if provided)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics
from data.dummy_data import (
    EXPENSE_CATEGORIES,
    MonthlyFinancials,
    TAX_RATE,
    COMPANY_NAME,
)


# ─── CFO KPI Dataclasses ──────────────────────────────────────────────────────

@dataclass
class PLStatement:
    """
    Complete Profit & Loss Statement for a period.
    All monetary values are in PKR.
    """
    period: str

    # ─── Revenue ──────────────────────────────────────
    revenue: float = 0.0
    revenue_growth_mom: float = 0.0      # Month-over-month %
    revenue_ytd: float = 0.0             # Year-to-date cumulative

    # ─── Cost of Goods Sold ───────────────────────────
    cogs: float = 0.0
    cogs_ratio: float = 0.0              # COGS / Revenue × 100

    # ─── Gross Profit ─────────────────────────────────
    gross_profit: float = 0.0
    gross_margin: float = 0.0            # Gross Profit / Revenue × 100

    # ─── Operating Expenses ───────────────────────────
    opex_total: float = 0.0
    opex_breakdown: Dict[str, float] = field(default_factory=dict)
    opex_ratio: float = 0.0              # Opex / Revenue × 100

    # ─── Operating Profit (EBIT) ──────────────────────
    ebit: float = 0.0
    ebit_margin: float = 0.0

    # ─── EBITDA (estimates depreciation at 1% of revenue) ──
    estimated_depreciation: float = 0.0
    ebitda: float = 0.0
    ebitda_margin: float = 0.0

    # ─── Tax ──────────────────────────────────────────
    tax_provision: float = 0.0
    effective_tax_rate: float = 0.0

    # ─── Net Profit (PAT) ─────────────────────────────
    net_profit: float = 0.0
    net_margin: float = 0.0
    net_profit_ytd: float = 0.0

    # ─── Liquidity Metrics ────────────────────────────
    burn_rate: float = 0.0               # Total monthly cash out
    total_expenses: float = 0.0

    # ─── Category Ratios ──────────────────────────────
    expense_ratios: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__

    def summary_lines(self) -> List[str]:
        """Returns formatted P&L summary for display/AI context."""
        sep = "─" * 50
        lines = [
            f"P&L STATEMENT — {self.period}",
            sep,
            f"Revenue (Net Sales)         PKR {self.revenue:>14,.0f}",
            f"Cost of Goods Sold          PKR {self.cogs:>14,.0f}",
            sep,
            f"GROSS PROFIT                PKR {self.gross_profit:>14,.0f}  ({self.gross_margin:.1f}%)",
            "",
            "Operating Expenses:",
        ]
        for cat, amount in sorted(self.opex_breakdown.items(), key=lambda x: -x[1]):
            label = EXPENSE_CATEGORIES.get(cat, {}).get("label", cat)
            lines.append(f"  {label:<35} PKR {amount:>10,.0f}")
        lines += [
            sep,
            f"Total OpEx                  PKR {self.opex_total:>14,.0f}",
            sep,
            f"EBIT (Operating Profit)     PKR {self.ebit:>14,.0f}  ({self.ebit_margin:.1f}%)",
            f"EBITDA                      PKR {self.ebitda:>14,.0f}  ({self.ebitda_margin:.1f}%)",
            f"Tax Provision ({TAX_RATE*100:.0f}%)       PKR {self.tax_provision:>14,.0f}",
            sep,
            f"NET PROFIT (PAT)            PKR {self.net_profit:>14,.0f}  ({self.net_margin:.1f}%)",
            sep,
        ]
        return lines


@dataclass
class AnnualSummary:
    """Annual aggregated financial summary."""
    year: int
    total_revenue: float
    total_cogs: float
    total_gross_profit: float
    avg_gross_margin: float
    total_opex: float
    total_ebit: float
    avg_ebit_margin: float
    total_tax: float
    total_net_profit: float
    avg_net_margin: float
    best_month: str
    worst_month: str
    avg_monthly_revenue: float
    revenue_volatility: float    # Std deviation of monthly revenue
    yoy_growth: Optional[float]  # Year-over-year growth %

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class CFOAlert:
    """
    Automated CFO alert/insight based on financial analysis.
    Used for the AI-powered insight section.
    """
    severity: str           # "critical" | "warning" | "info" | "positive"
    category: str           # e.g., "profitability", "cash_flow", "expense_control"
    title: str
    message: str
    metric_value: float
    threshold: float
    recommendation: str

    @property
    def icon(self) -> str:
        icons = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵",
            "positive": "🟢",
        }
        return icons.get(self.severity, "⚪")


# ─── Main P&L Engine ──────────────────────────────────────────────────────────

class PLEngine:
    """
    Core P&L calculation engine for SmartCFO.
    
    Implements CFO-grade financial logic including:
    - Statement generation from raw expense data
    - Ratio analysis and KPI computation
    - Variance analysis (MoM, YoY)
    - Automated anomaly detection and alerts
    - Trend analysis and forecasting
    """

    def __init__(self, tax_rate: float = TAX_RATE):
        self.tax_rate = tax_rate
        self._depreciation_rate = 0.01  # Estimate: 1% of revenue as depreciation

    # ─── Core P&L Builder ─────────────────────────────────────────────────────

    def build_pl_statement(
        self,
        month: str,
        revenue: float,
        expense_breakdown: Dict[str, float],
        prev_revenue: float = 0.0,
        revenue_ytd: float = 0.0,
        net_profit_ytd: float = 0.0,
    ) -> PLStatement:
        """
        Constructs a full P&L statement from revenue and expense inputs.
        
        Args:
            month:             Period label (e.g., "Jan 2024")
            revenue:           Total revenue for the period (PKR)
            expense_breakdown: Dict mapping category → amount (PKR)
            prev_revenue:      Previous period revenue for MoM calculation
            revenue_ytd:       Cumulative revenue year-to-date
            net_profit_ytd:    Cumulative net profit year-to-date
        
        Financial Logic:
            Gross Profit     = Revenue - COGS
            Gross Margin     = Gross Profit / Revenue × 100
            EBIT             = Gross Profit - Operating Expenses
            EBIT Margin      = EBIT / Revenue × 100
            EBITDA           = EBIT + Depreciation (estimated)
            Tax Provision    = max(0, EBIT × Tax Rate)
            Net Profit (PAT) = EBIT - Tax Provision
            Net Margin       = Net Profit / Revenue × 100
        """
        # Separate COGS from operating expenses
        cogs = expense_breakdown.get("cogs", 0.0)
        opex_breakdown = {k: v for k, v in expense_breakdown.items() if k != "cogs"}
        opex_total = sum(opex_breakdown.values())
        total_expenses = cogs + opex_total

        # ── Core P&L Calculations ──────────────────────────────────────────────
        gross_profit = revenue - cogs
        gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0

        ebit = gross_profit - opex_total
        ebit_margin = (ebit / revenue * 100) if revenue > 0 else 0.0

        # EBITDA: Add back estimated depreciation (non-cash charge)
        estimated_depreciation = revenue * self._depreciation_rate
        ebitda = ebit + estimated_depreciation
        ebitda_margin = (ebitda / revenue * 100) if revenue > 0 else 0.0

        # Pakistan Corporate Tax: Applied on taxable profit (EBIT basis)
        # Note: In reality, there are many adjustments. This is a simplified model.
        tax_provision = max(0.0, ebit * self.tax_rate) if ebit > 0 else 0.0
        effective_tax_rate = (tax_provision / ebit * 100) if ebit > 0 else 0.0

        net_profit = ebit - tax_provision
        net_margin = (net_profit / revenue * 100) if revenue > 0 else 0.0

        # ── Growth & Ratios ────────────────────────────────────────────────────
        revenue_growth_mom = (
            ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0.0
        )
        cogs_ratio = (cogs / revenue * 100) if revenue > 0 else 0.0
        opex_ratio = (opex_total / revenue * 100) if revenue > 0 else 0.0

        # Per-category expense ratios
        expense_ratios = {
            cat: (amt / revenue * 100) if revenue > 0 else 0.0
            for cat, amt in expense_breakdown.items()
        }

        return PLStatement(
            period=month,
            revenue=revenue,
            revenue_growth_mom=revenue_growth_mom,
            revenue_ytd=revenue_ytd + revenue,
            cogs=cogs,
            cogs_ratio=cogs_ratio,
            gross_profit=gross_profit,
            gross_margin=gross_margin,
            opex_total=opex_total,
            opex_breakdown=opex_breakdown,
            opex_ratio=opex_ratio,
            ebit=ebit,
            ebit_margin=ebit_margin,
            estimated_depreciation=estimated_depreciation,
            ebitda=ebitda,
            ebitda_margin=ebitda_margin,
            tax_provision=tax_provision,
            effective_tax_rate=effective_tax_rate,
            net_profit=net_profit,
            net_margin=net_margin,
            net_profit_ytd=net_profit_ytd + net_profit,
            burn_rate=total_expenses,
            total_expenses=total_expenses,
            expense_ratios=expense_ratios,
        )

    # ─── Process Full Year ────────────────────────────────────────────────────

    def process_year(self, monthly_financials: List[MonthlyFinancials]) -> List[PLStatement]:
        """
        Processes a full year of MonthlyFinancials into PLStatements.
        Handles YTD accumulators and MoM comparisons.
        """
        statements = []
        revenue_ytd = 0.0
        net_profit_ytd = 0.0
        prev_revenue = 0.0

        for i, mf in enumerate(monthly_financials):
            stmt = self.build_pl_statement(
                month=mf.month,
                revenue=mf.revenue,
                expense_breakdown=mf.expense_breakdown,
                prev_revenue=prev_revenue,
                revenue_ytd=revenue_ytd,
                net_profit_ytd=net_profit_ytd,
            )
            statements.append(stmt)
            revenue_ytd += mf.revenue
            net_profit_ytd += stmt.net_profit
            prev_revenue = mf.revenue

        return statements

    # ─── Annual Summary ───────────────────────────────────────────────────────

    def compute_annual_summary(
        self,
        statements: List[PLStatement],
        year: int,
    ) -> AnnualSummary:
        """
        Computes an annual financial summary from monthly P&L statements.
        Includes statistical analysis of revenue volatility.
        """
        revenues = [s.revenue for s in statements]
        net_profits = [s.net_profit for s in statements]
        gross_margins = [s.gross_margin for s in statements]
        ebit_margins = [s.ebit_margin for s in statements]
        net_margins = [s.net_margin for s in statements]

        best_month = statements[revenues.index(max(revenues))].period
        worst_month = statements[revenues.index(min(revenues))].period

        total_revenue = sum(revenues)
        total_cogs = sum(s.cogs for s in statements)
        total_gross_profit = sum(s.gross_profit for s in statements)
        total_opex = sum(s.opex_total for s in statements)
        total_ebit = sum(s.ebit for s in statements)
        total_tax = sum(s.tax_provision for s in statements)
        total_net_profit = sum(net_profits)

        revenue_volatility = statistics.stdev(revenues) if len(revenues) > 1 else 0.0

        return AnnualSummary(
            year=year,
            total_revenue=total_revenue,
            total_cogs=total_cogs,
            total_gross_profit=total_gross_profit,
            avg_gross_margin=statistics.mean(gross_margins),
            total_opex=total_opex,
            total_ebit=total_ebit,
            avg_ebit_margin=statistics.mean(ebit_margins),
            total_tax=total_tax,
            total_net_profit=total_net_profit,
            avg_net_margin=statistics.mean(net_margins),
            best_month=best_month,
            worst_month=worst_month,
            avg_monthly_revenue=statistics.mean(revenues),
            revenue_volatility=revenue_volatility,
            yoy_growth=None,  # Would require previous year data
        )

    # ─── Automated CFO Alerts ─────────────────────────────────────────────────

    def generate_cfo_alerts(self, stmt: PLStatement) -> List[CFOAlert]:
        """
        Generates automated CFO alerts based on financial thresholds.
        
        Threshold logic derived from Pakistani SME benchmarks:
        - IT services: Gross margin should be >45%
        - Net margin: Healthy is >10%, warning below 5%
        - Salary ratio: Should not exceed 30% of revenue
        - COGS ratio: Should be <35% for tech/services SME
        
        Returns list of CFOAlert objects, sorted by severity.
        """
        alerts = []

        # ── Gross Margin Alert ─────────────────────────────────────────────────
        if stmt.gross_margin < 40:
            alerts.append(CFOAlert(
                severity="critical",
                category="profitability",
                title="Low Gross Margin",
                message=f"Gross margin is {stmt.gross_margin:.1f}%, below the 40% benchmark for IT/tech SMEs.",
                metric_value=stmt.gross_margin,
                threshold=40.0,
                recommendation="Review COGS — consider renegotiating vendor contracts or shifting to higher-margin service lines.",
            ))
        elif stmt.gross_margin < 50:
            alerts.append(CFOAlert(
                severity="warning",
                category="profitability",
                title="Gross Margin Below Target",
                message=f"Gross margin of {stmt.gross_margin:.1f}% is below the 50% target.",
                metric_value=stmt.gross_margin,
                threshold=50.0,
                recommendation="Analyze COGS components. Consider pricing adjustments or product mix optimization.",
            ))
        else:
            alerts.append(CFOAlert(
                severity="positive",
                category="profitability",
                title="Healthy Gross Margin",
                message=f"Gross margin is {stmt.gross_margin:.1f}% — above target.",
                metric_value=stmt.gross_margin,
                threshold=50.0,
                recommendation="Maintain pricing discipline to sustain margins.",
            ))

        # ── Net Margin Alert ───────────────────────────────────────────────────
        if stmt.net_margin < 0:
            alerts.append(CFOAlert(
                severity="critical",
                category="profitability",
                title="Net Loss This Period",
                message=f"Negative net margin of {stmt.net_margin:.1f}%. The business is operating at a loss.",
                metric_value=stmt.net_margin,
                threshold=0.0,
                recommendation="Immediate expense review required. Identify and eliminate discretionary spending.",
            ))
        elif stmt.net_margin < 5:
            alerts.append(CFOAlert(
                severity="warning",
                category="profitability",
                title="Thin Net Margins",
                message=f"Net margin is {stmt.net_margin:.1f}%, leaving very little cushion.",
                metric_value=stmt.net_margin,
                threshold=5.0,
                recommendation="Focus on revenue growth and/or opex reduction. Review underperforming expense lines.",
            ))
        else:
            alerts.append(CFOAlert(
                severity="positive",
                category="profitability",
                title="Profitable Operations",
                message=f"Net margin of {stmt.net_margin:.1f}% indicates healthy profitability.",
                metric_value=stmt.net_margin,
                threshold=10.0,
                recommendation="Consider reinvesting profits into growth channels (marketing, R&D, talent).",
            ))

        # ── Salary Ratio Alert ─────────────────────────────────────────────────
        salary_ratio = stmt.expense_ratios.get("salaries", 0)
        if salary_ratio > 30:
            alerts.append(CFOAlert(
                severity="warning",
                category="expense_control",
                title="High Payroll Ratio",
                message=f"Salaries consuming {salary_ratio:.1f}% of revenue, above 30% benchmark.",
                metric_value=salary_ratio,
                threshold=30.0,
                recommendation="Review headcount efficiency. Consider performance-linked pay structures.",
            ))

        # ── Utilities Spike Alert ──────────────────────────────────────────────
        utilities_ratio = stmt.expense_ratios.get("utilities", 0)
        if utilities_ratio > 5:
            alerts.append(CFOAlert(
                severity="warning",
                category="expense_control",
                title="High Utility Costs (K-Electric)",
                message=f"Utilities at {utilities_ratio:.1f}% of revenue. K-Electric B-2 tariff is costly.",
                metric_value=utilities_ratio,
                threshold=5.0,
                recommendation="Investigate energy efficiency measures. Consider solar installation ROI analysis for Karachi offices.",
            ))

        # ── Revenue Growth Alert ───────────────────────────────────────────────
        if stmt.revenue_growth_mom < -10:
            alerts.append(CFOAlert(
                severity="critical",
                category="revenue",
                title="Sharp Revenue Decline",
                message=f"Revenue dropped {abs(stmt.revenue_growth_mom):.1f}% month-over-month.",
                metric_value=stmt.revenue_growth_mom,
                threshold=-10.0,
                recommendation="Investigate client churn, delayed payments, or seasonal factors. Review sales pipeline.",
            ))
        elif stmt.revenue_growth_mom > 15:
            alerts.append(CFOAlert(
                severity="positive",
                category="revenue",
                title="Strong Revenue Growth",
                message=f"Revenue grew {stmt.revenue_growth_mom:.1f}% month-over-month.",
                metric_value=stmt.revenue_growth_mom,
                threshold=15.0,
                recommendation="Excellent momentum. Ensure operational capacity scales with growth.",
            ))

        # ── COGS Ratio ─────────────────────────────────────────────────────────
        if stmt.cogs_ratio > 35:
            alerts.append(CFOAlert(
                severity="warning",
                category="expense_control",
                title="COGS Exceeding Target",
                message=f"COGS is {stmt.cogs_ratio:.1f}% of revenue, above 35% target for tech SMEs.",
                metric_value=stmt.cogs_ratio,
                threshold=35.0,
                recommendation="Review procurement contracts. Explore bulk purchasing or alternative vendors.",
            ))

        # Sort: critical first, then warning, info, positive
        severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

        return alerts

    # ─── Variance Analysis ────────────────────────────────────────────────────

    def compute_variance(
        self,
        current: PLStatement,
        previous: PLStatement,
    ) -> Dict[str, Dict]:
        """
        Computes period-over-period variance for key metrics.
        Returns absolute and percentage changes.
        
        Used for: MoM, QoQ comparisons in the dashboard.
        """
        def var(curr_val: float, prev_val: float) -> Dict:
            abs_change = curr_val - prev_val
            pct_change = (abs_change / abs(prev_val) * 100) if prev_val != 0 else 0.0
            return {
                "current": curr_val,
                "previous": prev_val,
                "absolute": abs_change,
                "percent": pct_change,
                "direction": "up" if abs_change > 0 else "down" if abs_change < 0 else "flat",
            }

        return {
            "revenue": var(current.revenue, previous.revenue),
            "gross_profit": var(current.gross_profit, previous.gross_profit),
            "gross_margin": var(current.gross_margin, previous.gross_margin),
            "ebit": var(current.ebit, previous.ebit),
            "net_profit": var(current.net_profit, previous.net_profit),
            "net_margin": var(current.net_margin, previous.net_margin),
            "total_expenses": var(current.total_expenses, previous.total_expenses),
            "burn_rate": var(current.burn_rate, previous.burn_rate),
        }

    # ─── Runway Calculator ────────────────────────────────────────────────────

    def compute_runway(
        self,
        cash_balance: float,
        statements: List[PLStatement],
        months_to_avg: int = 3,
    ) -> Dict:
        """
        Calculates cash runway based on recent burn rate.
        
        Args:
            cash_balance:   Current cash in bank (PKR)
            statements:     List of P&L statements (most recent last)
            months_to_avg:  Number of recent months to average burn rate
        
        Returns:
            Dict with runway months, burn rate, and survival date
        """
        if not statements:
            return {"runway_months": 0, "burn_rate": 0, "status": "insufficient_data"}

        recent = statements[-months_to_avg:]
        avg_burn = statistics.mean([s.burn_rate for s in recent])
        avg_revenue = statistics.mean([s.revenue for s in recent])
        net_burn = avg_burn - avg_revenue  # Negative = generating cash

        if net_burn <= 0:
            runway_months = float('inf')
            status = "cash_positive"
        else:
            runway_months = cash_balance / net_burn
            if runway_months > 24:
                status = "healthy"
            elif runway_months > 12:
                status = "adequate"
            elif runway_months > 6:
                status = "warning"
            else:
                status = "critical"

        return {
            "runway_months": runway_months,
            "avg_monthly_burn": avg_burn,
            "avg_monthly_revenue": avg_revenue,
            "net_monthly_burn": net_burn,
            "cash_balance": cash_balance,
            "status": status,
        }

    # ─── Forecasting (Simple Linear Regression) ───────────────────────────────

    def forecast_revenue(
        self,
        statements: List[PLStatement],
        periods_ahead: int = 3,
    ) -> List[Dict]:
        """
        Simple linear trend forecast for next N months.
        Uses ordinary least squares (OLS) regression on monthly revenue.
        
        Note: For production, consider ARIMA or Prophet for better accuracy.
        This is intentionally simple to run without heavy ML dependencies.
        """
        if len(statements) < 3:
            return []

        revenues = [s.revenue for s in statements]
        n = len(revenues)
        x = list(range(n))

        # OLS: calculate slope and intercept
        x_mean = sum(x) / n
        y_mean = sum(revenues) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, revenues))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean

        forecasts = []
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        last_month_idx = months_order.index(statements[-1].period) if statements[-1].period in months_order else 0

        for i in range(1, periods_ahead + 1):
            future_x = n - 1 + i
            forecasted_revenue = intercept + slope * future_x
            month_label = months_order[(last_month_idx + i) % 12]
            forecasts.append({
                "period": month_label,
                "forecasted_revenue": max(0, forecasted_revenue),
                "is_forecast": True,
            })

        return forecasts


# ─── Expense Categorizer (from OCR extraction) ────────────────────────────────

class ExpenseCategorizer:
    """
    Takes extracted invoice text and categorizes expenses into
    the P&L-compatible categories.
    
    Used in the OCR pipeline → P&L Engine flow.
    """

    # Keyword → category mapping (same as ocr_pipeline but more comprehensive)
    KEYWORD_MAP = {
        "k-electric": "utilities",
        "ke bill": "utilities",
        "sui gas": "utilities",
        "ssgc": "utilities",
        "electricity": "utilities",
        "power bill": "utilities",
        "kwh": "utilities",
        "stormfiber": "internet_telecom",
        "ptcl": "internet_telecom",
        "nayatel": "internet_telecom",
        "jazz": "internet_telecom",
        "zong": "internet_telecom",
        "internet": "internet_telecom",
        "fiber": "internet_telecom",
        "broadband": "internet_telecom",
        "rent": "rent",
        "lease": "rent",
        "tenancy": "rent",
        "payroll": "salaries",
        "salary": "salaries",
        "staff remuneration": "salaries",
        "eobi": "salaries",
        "sessi": "salaries",
        "office supplies": "supplies",
        "stationery": "supplies",
        "carrefour": "supplies",
        "al-fatah": "supplies",
        "naheed": "supplies",
        "metro cash": "supplies",
        "microsoft": "software",
        "adobe": "software",
        "zoom": "software",
        "slack": "software",
        "github": "software",
        "saas": "software",
        "subscription": "software",
        "meta platforms": "marketing",
        "facebook ads": "marketing",
        "google ads": "marketing",
        "digital marketing": "marketing",
        "advertising": "marketing",
        "bykea": "transport",
        "tcs": "transport",
        "leopards": "transport",
        "courier": "transport",
        "transport": "transport",
        "chartered accountant": "professional_fees",
        "legal": "professional_fees",
        "audit fee": "professional_fees",
        "tax advisory": "professional_fees",
        "it hardware": "cogs",
        "laptop": "cogs",
        "distributor": "cogs",
        "procurement": "cogs",
    }

    def categorize(self, text: str, amount: float) -> Tuple[str, float]:
        """
        Categorizes an expense from its text description.
        
        Returns:
            Tuple of (category, amount)
        """
        text_lower = text.lower()
        for keyword, category in self.KEYWORD_MAP.items():
            if keyword in text_lower:
                return category, amount
        return "uncategorized", amount

    def categorize_batch(
        self, extracted_invoices: list
    ) -> Dict[str, float]:
        """
        Processes a list of ExtractedInvoice objects and returns
        a Dict[category → total_amount] suitable for PLEngine.
        
        This is the bridge between OCR extraction and P&L calculation.
        """
        category_totals: Dict[str, float] = {}

        for invoice in extracted_invoices:
            cat = invoice.category
            if cat == "uncategorized":
                # Try categorizing from raw text
                cat, _ = self.categorize(invoice.raw_text, invoice.amount_total)

            if invoice.amount_total > 0:
                category_totals[cat] = category_totals.get(cat, 0) + invoice.amount_total

        return category_totals


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.dummy_data import generate_full_year_data

    data = generate_full_year_data(2024)
    engine = PLEngine()

    statements = engine.process_year(data["financials"])
    summary = engine.compute_annual_summary(statements, 2024)

    print("=" * 60)
    print(f"  SMARTCFO — P&L ENGINE TEST  |  {COMPANY_NAME}")
    print("=" * 60)

    # Print first 3 months
    for stmt in statements[:3]:
        print()
        print(f"\n📊 {stmt.period}:")
        for line in stmt.summary_lines():
            print(f"   {line}")

    print("\n" + "=" * 60)
    print("  📈 ANNUAL SUMMARY 2024")
    print("=" * 60)
    print(f"  Total Revenue   : PKR {summary.total_revenue:>15,.0f}")
    print(f"  Total Net Profit: PKR {summary.total_net_profit:>15,.0f}")
    print(f"  Avg Net Margin  : {summary.avg_net_margin:.1f}%")
    print(f"  Best Month      : {summary.best_month}")
    print(f"  Worst Month     : {summary.worst_month}")

    print("\n🔔 CFO Alerts (January):")
    alerts = engine.generate_cfo_alerts(statements[0])
    for alert in alerts:
        print(f"  {alert.icon} [{alert.severity.upper()}] {alert.title}: {alert.message}")
