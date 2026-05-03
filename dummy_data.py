"""
============================================================
SmartCFO — Dummy Data & Synthetic Receipt Generator
Pakistani SME Context | PKR Currency
============================================================

This module provides:
1. A realistic Pakistani SME financial dataset (PKR)
2. Synthetic PDF-like receipt text for OCR simulation
3. Monthly P&L data for visualization

Financial Logic (CFO Perspective):
- Gross Profit  = Revenue - COGS (Cost of Goods Sold)
- Operating Profit = Gross Profit - Operating Expenses
- Net Profit    = Operating Profit - Tax Provisions
- Profit Margin = (Net Profit / Revenue) × 100
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import date, timedelta
import json

# ─── Constants ────────────────────────────────────────────────────────────────

PKR = "PKR"
COMPANY_NAME = "TechBridge Solutions (Pvt.) Ltd."
COMPANY_NTN = "1234567-8"          # National Tax Number
COMPANY_ADDRESS = "Office 4B, Dolmen City, Block-4, Clifton, Karachi, Sindh 75600"
COMPANY_STRN = "03-16-9999-001-88" # Sales Tax Registration Number

# ─── Expense Categories (Pakistani SME) ───────────────────────────────────────

EXPENSE_CATEGORIES = {
    "utilities": {
        "label": "Utilities & Power",
        "vendors": ["K-Electric Limited", "Sui Southern Gas Company (SSGC)"],
        "color": "#EF4444",
    },
    "internet_telecom": {
        "label": "Internet & Telecom",
        "vendors": [
            "StormFiber (Cybernet)",
            "PTCL Broadband",
            "Nayatel Pvt Ltd",
            "Jazz Business",
            "Zong 4G Enterprise",
        ],
        "color": "#F97316",
    },
    "rent": {
        "label": "Office Rent",
        "vendors": ["Dolmen Mall Clifton — Property Mgmt"],
        "color": "#EAB308",
    },
    "salaries": {
        "label": "Salaries & Payroll",
        "vendors": ["Internal Payroll — HR Dept"],
        "color": "#22C55E",
    },
    "supplies": {
        "label": "Office Supplies",
        "vendors": [
            "M/s Al-Fatah General Store",
            "Carrefour Pakistan",
            "Naheed Super Market",
            "Metro Cash & Carry Pakistan",
        ],
        "color": "#3B82F6",
    },
    "software": {
        "label": "Software & Subscriptions",
        "vendors": [
            "Microsoft Pakistan (CSP Partner)",
            "Adobe Systems — APAC",
            "Zoom Video Communications",
            "Slack Technologies",
            "GitHub Enterprise",
        ],
        "color": "#8B5CF6",
    },
    "transport": {
        "label": "Transport & Logistics",
        "vendors": [
            "Bykea Business",
            "TCS Couriers Pvt Ltd",
            "M/s City Cab Service",
        ],
        "color": "#EC4899",
    },
    "marketing": {
        "label": "Marketing & Advertising",
        "vendors": [
            "Meta Platforms (Facebook Ads)",
            "Google Ads Pakistan",
            "Dawn Media Group",
        ],
        "color": "#14B8A6",
    },
    "professional_fees": {
        "label": "Professional Fees",
        "vendors": [
            "Rizvi, Isa & Co. — Chartered Accountants",
            "LexBridge Law Associates",
        ],
        "color": "#6366F1",
    },
    "cogs": {
        "label": "Cost of Goods Sold",
        "vendors": ["Various Hardware Vendors", "Local IT Distributors"],
        "color": "#DC2626",
    },
}

# ─── Monthly Revenue Baseline (PKR) ───────────────────────────────────────────

MONTHLY_REVENUE_BASE = {
    "Jan": 2_850_000,
    "Feb": 3_100_000,
    "Mar": 3_450_000,
    "Apr": 3_200_000,
    "May": 3_750_000,
    "Jun": 4_100_000,
    "Jul": 3_900_000,
    "Aug": 3_650_000,
    "Sep": 4_250_000,
    "Oct": 4_500_000,
    "Nov": 4_800_000,
    "Dec": 5_200_000,
}

# ─── Monthly Expense Ratios (% of revenue per category) ──────────────────────

EXPENSE_RATIOS = {
    "cogs": 0.28,           # 28% of revenue
    "salaries": 0.22,       # 22% of revenue
    "rent": 0.06,           # Fixed-ish (~6%)
    "utilities": 0.03,      # 3% of revenue
    "internet_telecom": 0.02,
    "software": 0.02,
    "marketing": 0.04,
    "supplies": 0.015,
    "transport": 0.01,
    "professional_fees": 0.005,
}

TAX_RATE = 0.29  # Corporate tax rate in Pakistan (Finance Act 2023)


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Receipt:
    """Represents a single financial receipt/invoice."""
    receipt_id: str
    date: str
    vendor: str
    category: str
    amount: float
    description: str
    invoice_number: str
    raw_text: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "date": self.date,
            "vendor": self.vendor,
            "category": self.category,
            "amount": self.amount,
            "description": self.description,
            "invoice_number": self.invoice_number,
        }


@dataclass
class MonthlyFinancials:
    """Full P&L for a given month."""
    month: str
    revenue: float
    cogs: float
    gross_profit: float
    gross_margin: float
    operating_expenses: float
    operating_profit: float
    operating_margin: float
    tax_provision: float
    net_profit: float
    net_margin: float
    expense_breakdown: Dict[str, float]

    def to_dict(self) -> dict:
        return self.__dict__


# ─── Synthetic Receipt Text Generator ────────────────────────────────────────

def _generate_receipt_text(
    vendor: str,
    category: str,
    amount: float,
    invoice_number: str,
    receipt_date: str,
    description: str,
) -> str:
    """
    Generates realistic OCR-like invoice text for a Pakistani vendor.
    Simulates what pdfplumber or Tesseract would extract from a real PDF.
    
    Engineering Note:
    Real invoices in Pakistan often follow FBR (Federal Board of Revenue)
    prescribed formats with NTN, STRN, and Sales Tax breakdowns.
    """
    sales_tax_amount = amount * 0.18 if category != "salaries" else 0.0
    subtotal = amount - sales_tax_amount

    # Simulate slight OCR artifacts for realism
    ocr_noise = random.choice(["", "", " ", "  "])  # occasional spacing errors

    templates = {
        "utilities": f"""
K-ELECTRIC LIMITED
KE — Karachi Electric Supply Company
Registered Office: KE HQ, Korangi, Karachi
NTN: {random.randint(1000000,9999999)}-{random.randint(0,9)} | STRN: 03-99-{random.randint(1000,9999)}-001-11

                    INVOICE / BILL OF SUPPLY
========================================================
Invoice No.   : {invoice_number}
Bill Date     : {receipt_date}
Consumer No.  : KE-{random.randint(100000,999999)}
Account Title : {COMPANY_NAME}
Address       : {COMPANY_ADDRESS}

BILLING DETAILS:
Units Consumed (kWh)    :  {random.randint(800,2500):,}
Tariff Category         :  Commercial (B-2)
Energy Charges          :  PKR {subtotal:>12,.2f}
Fuel Price Adjustment   :  PKR {random.randint(500,2000):>12,.2f}
Meter Rent              :  PKR {random.randint(50,200):>12,.2f}
                         ─────────────────
Sub Total               :  PKR {subtotal:>12,.2f}
General Sales Tax (18%) :  PKR {sales_tax_amount:>12,.2f}
                         ─────────────────
TOTAL AMOUNT DUE{ocr_noise}        :  PKR {amount:>12,.2f}
========================================================
Due Date: {receipt_date}  |  Pay before due date to avoid surcharge.
FBR POS Invoice | Thank you for paying on time.
""",

        "internet_telecom": f"""
{vendor.upper()}
Pakistan Telecommunication Authority (PTA) Licensed Operator
NTN: {random.randint(1000000,9999999)}-{random.randint(0,9)}

                TAX INVOICE
============================================
Invoice No.     : {invoice_number}
Invoice Date    : {receipt_date}
Customer ID     : CUST-{random.randint(10000,99999)}
Customer Name   : {COMPANY_NAME}
Package         : {description}

SERVICE CHARGES:
Monthly Service Fee     : PKR {subtotal:>10,.2f}
Static IP Charges       : PKR {random.randint(500,1500):>10,.2f}
                         ──────────────
Sub-Total               : PKR {subtotal:>10,.2f}
Sales Tax @ 18%         : PKR {sales_tax_amount:>10,.2f}
WHT Deducted @ 10%      : PKR {-(amount*0.10):>10,.2f}
                         ──────────────
NET PAYABLE             : PKR {amount:>10,.2f}
============================================
Bank: MCB Bank Ltd | A/C: 0123456789
IBAN: PK36MUCB0000000123456789
""",

        "rent": f"""
LEASE / RENT RECEIPT
============================================
Landlord        : Dolmen City Management Pvt. Ltd.
Property        : Office Suite 4B, Block-4
                  Dolmen City, Clifton, Karachi
NTN             : {random.randint(1000000,9999999)}-{random.randint(0,9)}

Receipt No.     : {invoice_number}
Date            : {receipt_date}
Tenant          : {COMPANY_NAME}
NTN (Tenant)    : {COMPANY_NTN}

PAYMENT DETAILS:
Monthly Rent                : PKR {subtotal:>10,.2f}
Maintenance Charges         : PKR {random.randint(5000,15000):>10,.2f}
                             ──────────────
Sub-Total                   : PKR {subtotal:>10,.2f}
Withholding Tax (Section 155): PKR {amount*0.15:>10,.2f}
                             ──────────────
GROSS RENT PAID             : PKR {amount:>10,.2f}
============================================
Paid Via: Online Transfer | Ref: {random.randint(100000000,999999999)}
This receipt is computer-generated and valid without signature.
""",

        "supplies": f"""
{vendor}
General Merchants & Retailers
NTN: {random.randint(1000000,9999999)}-{random.randint(0,9)}

                    SALES RECEIPT
================================================
Receipt No.  : {invoice_number}
Date         : {receipt_date}
Sold To      : {COMPANY_NAME}
NTN (Buyer)  : {COMPANY_NTN}

ITEMS PURCHASED:
------------------------------------------------
Item                     Qty    Unit    Amount
------------------------------------------------
A4 Paper (Reams)          10   500.00    5,000
Printer Toner Cartridge    2  3,500.00    7,000
Ballpoint Pens (Box)       5   250.00    1,250
Whiteboard Markers         3   350.00    1,050
Stapler & Pins Set         2   600.00    1,200
Hand Sanitizer (500ml)     6   280.00    1,680
Face Masks (Box)           4   450.00    1,800
Miscellaneous Supplies     -        -    {subtotal - 19000:,.0f}
------------------------------------------------
Sub-Total                          : PKR {subtotal:>8,.2f}
GST @ 17%                          : PKR {sales_tax_amount:>8,.2f}
------------------------------------------------
TOTAL                              : PKR {amount:>8,.2f}
================================================
FBR Integrated POS System | Fiscal Invoice
""",

        "software": f"""
{vendor}
Software & Cloud Services Invoice
Billing Entity: APAC / Middle East Region

TAX INVOICE
============================================
Invoice #       : {invoice_number}
Invoice Date    : {receipt_date}
Bill To         :
  {COMPANY_NAME}
  {COMPANY_ADDRESS}
  NTN: {COMPANY_NTN}

SUBSCRIPTION DETAILS:
Product         : {description}
Billing Cycle   : Monthly
Seats / Licenses: {random.randint(5,25)}

                    Amount (USD)  Rate       PKR
Annual License :  ${random.randint(50,200)}.00  x 280.50  {subtotal:>10,.2f}
                                          ──────────
Sub-Total                                 {subtotal:>10,.2f}
Sales Tax (18%)                           {sales_tax_amount:>10,.2f}
                                          ──────────
TOTAL DUE (PKR)                           {amount:>10,.2f}
============================================
Paid via: Corporate Visa **** 4521
""",

        "marketing": f"""
{vendor}
Digital Advertising — Invoice

INVOICE
============================================
Invoice No.     : {invoice_number}
Date            : {receipt_date}
Advertiser      : {COMPANY_NAME}
Account ID      : {random.randint(100000000,999999999)}

CAMPAIGN SUMMARY:
Campaign Name   : {description}
Campaign Period : {receipt_date} (Monthly)
Ad Spend        : PKR {subtotal:,.2f}

BILLING:
Ad Spend (Net)          : PKR {subtotal:>10,.2f}
Platform Fee (2%)       : PKR {subtotal*0.02:>10,.2f}
Sales Tax @ 18%         : PKR {sales_tax_amount:>10,.2f}
                         ──────────────
TOTAL CHARGED           : PKR {amount:>10,.2f}
============================================
Auto-charged to: Corporate Card **** 7832
""",

        "professional_fees": f"""
{vendor}
CHARTERED ACCOUNTANTS & TAX CONSULTANTS
ICAP Registration No.: {random.randint(1000,9999)}

                    PROFESSIONAL FEE NOTE
======================================================
Fee Note No.    : {invoice_number}
Date            : {receipt_date}
Client          : {COMPANY_NAME}
NTN (Client)    : {COMPANY_NTN}

SERVICES RENDERED:
{description}

FEE BREAKDOWN:
Professional Fee                : PKR {subtotal:>10,.2f}
Out-of-pocket expenses          : PKR {random.randint(2000,5000):>10,.2f}
                                 ──────────────
Sub-Total                       : PKR {subtotal:>10,.2f}
Sales Tax @ 13% (Exempt if <5L) : PKR {sales_tax_amount:>10,.2f}
WHT Deductible u/s 153(1)(b)    : PKR {-(amount*0.10):>10,.2f}
                                 ──────────────
NET PAYABLE                     : PKR {amount:>10,.2f}
======================================================
Bank: HBL | A/C: {COMPANY_NTN.replace('-','')}
""",

        "transport": f"""
{vendor}
Transport & Logistics Services
NTN: {random.randint(1000000,9999999)}-{random.randint(0,9)}

DELIVERY / TRANSPORT INVOICE
============================================
Invoice No.  : {invoice_number}
Date         : {receipt_date}
Client       : {COMPANY_NAME}

TRIP DETAILS:
{description}

Trips / Deliveries  : {random.randint(15,60)}
Base Fare           : PKR {subtotal*0.85:>8,.2f}
Waiting Charges     : PKR {subtotal*0.10:>8,.2f}
Fuel Surcharge      : PKR {subtotal*0.05:>8,.2f}
                     ──────────
Sub-Total           : PKR {subtotal:>8,.2f}
GST @ 18%           : PKR {sales_tax_amount:>8,.2f}
                     ──────────
TOTAL               : PKR {amount:>8,.2f}
============================================
""",

        "cogs": f"""
M/S AL-HASSAN IT DISTRIBUTORS
Authorized Distributors — Technology Products
NTN: {random.randint(1000000,9999999)}-{random.randint(0,9)} | STRN: 03-14-{random.randint(1000,9999)}-001-55

                    TAX INVOICE
========================================================
Invoice No.     : {invoice_number}
Date            : {receipt_date}
Sold To         : {COMPANY_NAME}
NTN             : {COMPANY_NTN}

GOODS SUPPLIED:
Description                         Qty  Unit Price      Total
--------------------------------------------------------------
Dell Laptop (i5, 8GB, 256SSD)         2   95,000     190,000
HP Wireless Keyboard+Mouse Kit         5    3,500      17,500
Samsung Monitor 24"                    2   35,000      70,000
Networking Cables (Cat6, 50m)          4    2,800      11,200
UPS 1000VA APC                         1   18,000      18,000
Miscellaneous IT Hardware              -        -      {subtotal - 306700:,.0f}
                                              ──────────────
Sub-Total                                     PKR {subtotal:>12,.2f}
Sales Tax @ 17%                               PKR {sales_tax_amount:>12,.2f}
                                              ──────────────
INVOICE TOTAL                                 PKR {amount:>12,.2f}
========================================================
E&OE. Goods once sold will not be taken back.
FBR Integrated POS — Fiscal Invoice #{random.randint(1000000,9999999)}
""",

        "salaries": f"""
{COMPANY_NAME}
PAYROLL DISBURSEMENT ADVICE
NTN: {COMPANY_NTN}

                MONTHLY SALARY STATEMENT
============================================
Pay Period      : {receipt_date}
Reference No.   : {invoice_number}
Prepared By     : HR & Payroll Department

PAYROLL SUMMARY:
Total Headcount             : {random.randint(12,18)}

Gross Salaries              : PKR {amount*1.15:>10,.2f}
EOBI Contribution (Employer): PKR {amount*0.05:>10,.2f}
SESSI Contribution          : PKR {amount*0.03:>10,.2f}
                             ──────────────
Total Payroll Cost          : PKR {amount*1.18:>10,.2f}

DEDUCTIONS:
Income Tax (Staff)          : PKR {amount*0.08:>10,.2f}
EOBI (Employee Share)       : PKR {amount*0.01:>10,.2f}
Provident Fund              : PKR {amount*0.05:>10,.2f}
                             ──────────────
NET DISBURSED               : PKR {amount:>10,.2f}
============================================
Paid Via: Bank Transfer (Payroll Account)
Bank: MCB Bank | Batch Ref: PAY-{random.randint(10000,99999)}
Authorized by: CEO & CFO
""",
    }

    # Default fallback template
    return templates.get(
        category,
        f"""
INVOICE
Vendor: {vendor}
Invoice No: {invoice_number}
Date: {receipt_date}
Description: {description}
Amount: PKR {amount:,.2f}
""",
    )


# ─── Dummy Receipt Generator ──────────────────────────────────────────────────

def generate_monthly_receipts(month: str, year: int = 2024) -> List[Receipt]:
    """
    Generates a realistic set of expense receipts for a Pakistani SME
    for a given month. Each receipt includes raw OCR-like text.
    
    Args:
        month: Three-letter month abbreviation (e.g., "Jan")
        year:  Fiscal year
    
    Returns:
        List of Receipt objects with full simulated invoice text
    """
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    month_num = month_map.get(month, 1)
    revenue = MONTHLY_REVENUE_BASE[month]

    receipts = []
    receipt_counter = 1

    for category, ratio in EXPENSE_RATIOS.items():
        # Apply ±10% random variation to make data more realistic
        variance = random.uniform(0.90, 1.10)
        base_amount = revenue * ratio * variance

        cat_data = EXPENSE_CATEGORIES[category]
        vendor = random.choice(cat_data["vendors"])

        # Generate invoice date within the month
        day = random.randint(1, 28)
        receipt_date = f"{year}-{month_num:02d}-{day:02d}"

        invoice_number = f"INV-{year}{month_num:02d}-{receipt_counter:04d}"
        receipt_id = f"RCP-{month}{year}-{receipt_counter:03d}"

        # Category-specific descriptions
        descriptions = {
            "utilities": "Monthly electricity bill — Commercial B-2 Tariff",
            "internet_telecom": "Business Fiber 100 Mbps + Static IP Bundle",
            "rent": f"Monthly office rent — {month} {year}",
            "salaries": f"Monthly payroll disbursement — {month} {year}",
            "supplies": f"Office supplies procurement — {month} {year}",
            "software": "Microsoft 365 Business + Azure Credits",
            "marketing": f"Digital marketing campaign — {month} {year}",
            "professional_fees": f"Monthly accounting & tax advisory — {month} {year}",
            "transport": f"Staff transport & courier services — {month} {year}",
            "cogs": f"IT hardware & product procurement — {month} {year}",
        }

        description = descriptions.get(category, f"{cat_data['label']} — {month} {year}")

        raw_text = _generate_receipt_text(
            vendor=vendor,
            category=category,
            amount=round(base_amount, 2),
            invoice_number=invoice_number,
            receipt_date=receipt_date,
            description=description,
        )

        receipt = Receipt(
            receipt_id=receipt_id,
            date=receipt_date,
            vendor=vendor,
            category=category,
            amount=round(base_amount, 2),
            description=description,
            invoice_number=invoice_number,
            raw_text=raw_text,
        )
        receipts.append(receipt)
        receipt_counter += 1

    return receipts


# ─── Full Year Dataset ─────────────────────────────────────────────────────────

def generate_full_year_data(year: int = 2024) -> Dict:
    """
    Generates a complete 12-month financial dataset for the SME.
    
    Returns:
        Dictionary with monthly financials and all receipts
    """
    all_months = list(MONTHLY_REVENUE_BASE.keys())
    all_financials = []
    all_receipts = {}

    for month in all_months:
        revenue = MONTHLY_REVENUE_BASE[month] * random.uniform(0.95, 1.05)
        receipts = generate_monthly_receipts(month, year)

        # Sum expenses by category
        expense_breakdown = {}
        for r in receipts:
            expense_breakdown[r.category] = expense_breakdown.get(r.category, 0) + r.amount

        cogs = expense_breakdown.get("cogs", 0)
        operating_expenses = sum(v for k, v in expense_breakdown.items() if k != "cogs")
        gross_profit = revenue - cogs
        gross_margin = (gross_profit / revenue) * 100
        operating_profit = gross_profit - operating_expenses
        operating_margin = (operating_profit / revenue) * 100
        tax_provision = max(0, operating_profit * TAX_RATE)
        net_profit = operating_profit - tax_provision
        net_margin = (net_profit / revenue) * 100

        mf = MonthlyFinancials(
            month=month,
            revenue=round(revenue, 2),
            cogs=round(cogs, 2),
            gross_profit=round(gross_profit, 2),
            gross_margin=round(gross_margin, 2),
            operating_expenses=round(operating_expenses, 2),
            operating_profit=round(operating_profit, 2),
            operating_margin=round(operating_margin, 2),
            tax_provision=round(tax_provision, 2),
            net_profit=round(net_profit, 2),
            net_margin=round(net_margin, 2),
            expense_breakdown={k: round(v, 2) for k, v in expense_breakdown.items()},
        )
        all_financials.append(mf)
        all_receipts[month] = receipts

    return {
        "company": COMPANY_NAME,
        "year": year,
        "financials": all_financials,
        "receipts": all_receipts,
        "tax_rate": TAX_RATE,
    }


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = generate_full_year_data(2024)
    print(f"✅ Generated data for: {data['company']}")
    print(f"   Months covered: {len(data['financials'])}")
    jan = data["financials"][0]
    print(f"\n📊 January 2024 Snapshot:")
    print(f"   Revenue       : PKR {jan.revenue:>12,.0f}")
    print(f"   Gross Profit  : PKR {jan.gross_profit:>12,.0f} ({jan.gross_margin:.1f}%)")
    print(f"   Net Profit    : PKR {jan.net_profit:>12,.0f} ({jan.net_margin:.1f}%)")
    print(f"\n📄 Sample Receipt (Jan):")
    print(data["receipts"]["Jan"][0].raw_text[:400])
