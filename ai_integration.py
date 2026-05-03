"""
============================================================
SmartCFO — AI Integration Layer
Groq (LLaMA) + Gemini API for Intelligent Receipt Analysis
============================================================

This module implements AI-powered receipt processing using:
- Groq API  → Ultra-fast LLaMA 3.1 inference (primary)
- Gemini API → Google's multimodal model (fallback / image receipts)

Architecture:
┌─────────────────────────────────────────────┐
│  Receipt Text / Image                        │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────┐                        │
│  │  AI Router      │  ← Try Groq first      │
│  │  (with retry)   │  ← Fallback to Gemini  │
│  └─────────────────┘                        │
│         │                                    │
│         ▼                                    │
│  Structured JSON extraction                  │
│  (amount, vendor, date, category, tax)       │
│         │                                    │
│         ▼                                    │
│  Validation & Confidence Scoring             │
│         │                                    │
│         ▼                                    │
│  CFO Insights Generation                     │
└─────────────────────────────────────────────┘

Prompt Engineering Principles (Pakistani SME Context):
1. Explicit currency format instructions (PKR, South Asian notation)
2. FBR invoice structure awareness (NTN, STRN, Sales Tax)
3. Pakistani vendor name normalization
4. Confidence calibration for OCR noise
5. Multi-turn conversation for complex invoices
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ─── Try importing AI SDKs gracefully ────────────────────────────────────────

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("groq not installed. Run: pip install groq")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")

try:
    import requests as _requests_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class AIExtractedData:
    """
    Structured data returned by AI receipt extraction.
    This is the AI-powered complement to the regex-based ExtractedInvoice.
    """
    # Core financials
    vendor_name: str = ""
    vendor_normalized: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    amount_total: float = 0.0
    amount_subtotal: float = 0.0
    amount_tax: float = 0.0
    tax_rate_applied: float = 0.0
    currency: str = "PKR"

    # Classification
    expense_category: str = "uncategorized"
    expense_subcategory: str = ""
    is_recurring: bool = False
    payment_method: str = ""

    # AI metadata
    model_used: str = ""
    extraction_confidence: float = 0.0
    ai_notes: str = ""
    raw_ai_response: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.amount_total > 0 and self.vendor_name != ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CFOInsight:
    """AI-generated CFO insight for a period's financial data."""
    summary: str = ""
    key_risks: List[str] = field(default_factory=list)
    cost_optimization_tips: List[str] = field(default_factory=list)
    revenue_observations: str = ""
    cashflow_outlook: str = ""
    tax_notes: str = ""
    model_used: str = ""
    raw_response: str = ""


# ─── System Prompts ───────────────────────────────────────────────────────────

RECEIPT_EXTRACTION_SYSTEM_PROMPT = """You are a Senior Pakistani CFO and AI data extraction specialist with deep expertise in:
- Pakistani tax law (FBR regulations, Sales Tax Act 1990, Income Tax Ordinance 2001)
- Local vendor ecosystem (K-Electric, PTCL, StormFiber, Sui Gas, etc.)
- Pakistani invoice formats (NTN, STRN, FBR POS integration)
- South Asian number formatting (Lakhs, Crores, PKR notation)

Your task is to extract structured financial data from invoice/receipt text.

CRITICAL RULES:
1. Always return ONLY valid JSON — no markdown, no explanations, no backticks
2. All monetary amounts MUST be in PKR (Pakistani Rupees) as floating point numbers
3. Pakistani number format: 1,23,456.00 → extract as 123456.00
4. "Total Amount Due", "Net Payable", "Grand Total" = the final payable amount
5. If Sales Tax is 18%, the pre-tax amount = total / 1.18
6. For Withholding Tax (WHT): the net payable = gross amount - WHT deduction
7. Dates: normalize to YYYY-MM-DD format
8. Confidence: 0.0 to 1.0 (1.0 = perfect extraction, 0.5 = some uncertainty)

EXPENSE CATEGORIES (use these exact strings):
- utilities        → K-Electric, SSGC gas, water bills
- internet_telecom → StormFiber, PTCL, Nayatel, Jazz, Zong
- rent             → Office/shop rent payments
- salaries         → Payroll, EOBI, staff remuneration
- supplies         → Office supplies, stationery, groceries
- software         → SaaS subscriptions, Microsoft, Adobe, Zoom
- marketing        → Facebook Ads, Google Ads, print media
- transport        → Bykea, TCS, couriers, cab services
- professional_fees → CA fees, legal fees, consultancy
- cogs             → Hardware, IT equipment, raw materials
- uncategorized    → Cannot determine category"""

RECEIPT_EXTRACTION_USER_TEMPLATE = """Extract structured data from this Pakistani invoice/receipt:

---INVOICE TEXT START---
{invoice_text}
---INVOICE TEXT END---

Return ONLY this JSON structure (no other text):
{{
  "vendor_name": "exact vendor name from invoice",
  "vendor_normalized": "standardized vendor name",
  "invoice_number": "invoice/receipt number",
  "invoice_date": "YYYY-MM-DD",
  "amount_total": 0.00,
  "amount_subtotal": 0.00,
  "amount_tax": 0.00,
  "tax_rate_applied": 0.0,
  "currency": "PKR",
  "expense_category": "category from the list",
  "expense_subcategory": "more specific subcategory",
  "is_recurring": false,
  "payment_method": "bank_transfer|cash|card|cheque|unknown",
  "extraction_confidence": 0.0,
  "ai_notes": "any important observations or ambiguities"
}}"""

CFO_INSIGHTS_SYSTEM_PROMPT = """You are a Pakistani CFO with 20+ years of experience advising SMEs in Karachi.
You understand the local business environment including:
- FBR tax regulations and compliance requirements
- SECP filing requirements for private limited companies
- State Bank of Pakistan (SBP) forex and banking regulations  
- Karachi business landscape (K-Electric outages impact, local market conditions)
- Pakistan's economic conditions (inflation, PKR depreciation, interest rates)

Provide concise, actionable CFO insights. Be direct and practical.
Always reference PKR amounts and local Pakistani context.
Format your response as JSON only."""

CFO_INSIGHTS_USER_TEMPLATE = """Analyze this Pakistani SME's monthly P&L data and provide CFO insights:

COMPANY: {company_name}
PERIOD: {period}

P&L SUMMARY:
Revenue:          PKR {revenue:,.0f}
COGS:             PKR {cogs:,.0f}
Gross Profit:     PKR {gross_profit:,.0f} ({gross_margin:.1f}%)
Total OpEx:       PKR {opex_total:,.0f}
EBIT:             PKR {ebit:,.0f} ({ebit_margin:.1f}%)
Tax Provision:    PKR {tax_provision:,.0f}
Net Profit:       PKR {net_profit:,.0f} ({net_margin:.1f}%)

EXPENSE BREAKDOWN:
{expense_details}

MoM REVENUE CHANGE: {revenue_growth:.1f}%

Return ONLY this JSON (no other text):
{{
  "summary": "2-3 sentence executive summary in CFO tone",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "cost_optimization_tips": ["actionable tip 1", "actionable tip 2", "actionable tip 3"],
  "revenue_observations": "analysis of revenue trend",
  "cashflow_outlook": "30-60-90 day cashflow perspective",
  "tax_notes": "relevant FBR/tax compliance notes for Pakistan"
}}"""


# ─── Groq Client ──────────────────────────────────────────────────────────────

class GroqReceiptExtractor:
    """
    Extracts structured data from invoice text using Groq's ultra-fast
    LLaMA inference API.
    
    Why Groq?
    - ~400 tokens/second inference speed (vs ~50 for standard APIs)
    - Free tier: 30 requests/min, 14,400 requests/day
    - LLaMA 3.1 70B has strong JSON extraction capability
    - Low latency critical for real-time receipt processing UX
    """

    # Available Groq models (as of 2024)
    MODELS = {
        "fast": "llama-3.1-8b-instant",     # Fastest, less accurate
        "balanced": "llama-3.1-70b-versatile",  # Best for financial extraction
        "smart": "mixtral-8x7b-32768",       # Good for complex invoices
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "balanced"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = self.MODELS.get(model, model)
        self.client = None

        if self.api_key and HAS_GROQ:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Groq client initialization failed: {e}")

    @property
    def is_available(self) -> bool:
        return bool(self.client and self.api_key)

    def extract_receipt(
        self,
        invoice_text: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AIExtractedData:
        """
        Extracts structured data from invoice text via Groq API.
        
        Implements exponential backoff retry logic for rate limit handling.
        
        Args:
            invoice_text: Raw OCR/PDF text of the invoice
            max_retries:  Number of retries on failure
            retry_delay:  Base delay between retries (seconds)
        
        Returns:
            AIExtractedData with extracted financial fields
        """
        if not self.is_available:
            return AIExtractedData(
                errors=["Groq API not configured. Set GROQ_API_KEY environment variable."],
                model_used="none",
            )

        # Truncate very long invoices (token limit management)
        truncated_text = invoice_text[:4000] if len(invoice_text) > 4000 else invoice_text

        messages = [
            {"role": "system", "content": RECEIPT_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RECEIPT_EXTRACTION_USER_TEMPLATE.format(
                    invoice_text=truncated_text
                ),
            },
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,       # Low temperature for consistent extraction
                    max_tokens=800,
                    response_format={"type": "json_object"},  # Force JSON output
                )

                raw_response = response.choices[0].message.content
                return self._parse_response(raw_response, self.model_name)

            except Exception as e:
                last_error = str(e)
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Groq API error on attempt {attempt + 1}: {e}")
                    time.sleep(retry_delay)

        return AIExtractedData(
            errors=[f"Groq extraction failed after {max_retries} attempts: {last_error}"],
            model_used=self.model_name,
        )

    def generate_cfo_insights(self, pl_data: dict) -> CFOInsight:
        """
        Generates CFO insights from P&L data using Groq.
        
        Args:
            pl_data: Dictionary with P&L metrics (from PLStatement.to_dict())
        
        Returns:
            CFOInsight with AI-generated analysis
        """
        if not self.is_available:
            return CFOInsight(
                summary="AI insights unavailable — Groq API key not configured.",
                model_used="none",
            )

        # Format expense breakdown for the prompt
        expense_details = "\n".join([
            f"  {k}: PKR {v:,.0f}"
            for k, v in pl_data.get("opex_breakdown", {}).items()
        ])

        user_message = CFO_INSIGHTS_USER_TEMPLATE.format(
            company_name=pl_data.get("company_name", "TechBridge Solutions"),
            period=pl_data.get("period", ""),
            revenue=pl_data.get("revenue", 0),
            cogs=pl_data.get("cogs", 0),
            gross_profit=pl_data.get("gross_profit", 0),
            gross_margin=pl_data.get("gross_margin", 0),
            opex_total=pl_data.get("opex_total", 0),
            ebit=pl_data.get("ebit", 0),
            ebit_margin=pl_data.get("ebit_margin", 0),
            tax_provision=pl_data.get("tax_provision", 0),
            net_profit=pl_data.get("net_profit", 0),
            net_margin=pl_data.get("net_margin", 0),
            expense_details=expense_details,
            revenue_growth=pl_data.get("revenue_growth_mom", 0),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CFO_INSIGHTS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.4,    # Slightly higher temp for creative insights
                max_tokens=1200,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            return CFOInsight(
                summary=parsed.get("summary", ""),
                key_risks=parsed.get("key_risks", []),
                cost_optimization_tips=parsed.get("cost_optimization_tips", []),
                revenue_observations=parsed.get("revenue_observations", ""),
                cashflow_outlook=parsed.get("cashflow_outlook", ""),
                tax_notes=parsed.get("tax_notes", ""),
                model_used=self.model_name,
                raw_response=raw,
            )

        except Exception as e:
            logger.error(f"CFO insights generation failed: {e}")
            return CFOInsight(
                summary=f"AI insight generation failed: {str(e)}",
                model_used=self.model_name,
            )

    def _parse_response(self, raw_response: str, model_name: str) -> AIExtractedData:
        """
        Parses and validates the JSON response from the AI model.
        Includes robust error handling for malformed JSON.
        """
        raw_clean = raw_response.strip()

        # Strip markdown code fences if present (model sometimes ignores json_object mode)
        raw_clean = re.sub(r'^```(?:json)?\s*', '', raw_clean)
        raw_clean = re.sub(r'\s*```$', '', raw_clean)

        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw response: {raw_clean[:200]}")
            return AIExtractedData(
                errors=[f"JSON parsing failed: {str(e)}"],
                raw_ai_response=raw_clean,
                model_used=model_name,
            )

        # Validate and coerce types
        def safe_float(val, default=0.0) -> float:
            try:
                return float(str(val).replace(",", "").replace("PKR", "").strip())
            except (ValueError, TypeError):
                return default

        def safe_bool(val, default=False) -> bool:
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "yes", "1")

        return AIExtractedData(
            vendor_name=str(data.get("vendor_name", "")),
            vendor_normalized=str(data.get("vendor_normalized", data.get("vendor_name", ""))),
            invoice_number=str(data.get("invoice_number", "")),
            invoice_date=str(data.get("invoice_date", "")),
            amount_total=safe_float(data.get("amount_total", 0)),
            amount_subtotal=safe_float(data.get("amount_subtotal", 0)),
            amount_tax=safe_float(data.get("amount_tax", 0)),
            tax_rate_applied=safe_float(data.get("tax_rate_applied", 0)),
            currency=str(data.get("currency", "PKR")),
            expense_category=str(data.get("expense_category", "uncategorized")),
            expense_subcategory=str(data.get("expense_subcategory", "")),
            is_recurring=safe_bool(data.get("is_recurring", False)),
            payment_method=str(data.get("payment_method", "unknown")),
            extraction_confidence=safe_float(data.get("extraction_confidence", 0.5)),
            ai_notes=str(data.get("ai_notes", "")),
            raw_ai_response=raw_clean,
            model_used=model_name,
        )


# ─── Gemini Client ────────────────────────────────────────────────────────────

class GeminiReceiptExtractor:
    """
    Alternative extractor using Google's Gemini API.
    
    Why Gemini?
    - Supports image input → can process scanned receipts directly
    - Strong multilingual support (Urdu invoices)
    - Free tier available via Google AI Studio
    - Gemini Flash is very fast for text extraction tasks
    """

    MODELS = {
        "flash": "gemini-1.5-flash",       # Fast, free tier
        "pro": "gemini-1.5-pro",           # Best accuracy
        "flash_8b": "gemini-1.5-flash-8b", # Fastest, most economical
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = self.MODELS.get(model, model)
        self.model = None

        if self.api_key and HAS_GEMINI:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                    system_instruction=RECEIPT_EXTRACTION_SYSTEM_PROMPT,
                )
                logger.info(f"Gemini client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Gemini client initialization failed: {e}")

    @property
    def is_available(self) -> bool:
        return bool(self.model and self.api_key)

    def extract_receipt(self, invoice_text: str) -> AIExtractedData:
        """
        Extracts structured data using Gemini API.
        Also supports image uploads for scanned receipts.
        """
        if not self.is_available:
            return AIExtractedData(
                errors=["Gemini API not configured. Set GEMINI_API_KEY environment variable."],
                model_used="none",
            )

        prompt = RECEIPT_EXTRACTION_USER_TEMPLATE.format(
            invoice_text=invoice_text[:5000]
        )

        try:
            response = self.model.generate_content(prompt)
            raw = response.text
            # Reuse Groq's parser since JSON structure is identical
            groq_extractor = GroqReceiptExtractor.__new__(GroqReceiptExtractor)
            return groq_extractor._parse_response(raw, self.model_name)

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return AIExtractedData(
                errors=[f"Gemini extraction failed: {str(e)}"],
                model_used=self.model_name,
            )

    def extract_receipt_from_image(
        self, image_bytes: bytes, media_type: str = "image/jpeg"
    ) -> AIExtractedData:
        """
        Extracts data from a scanned receipt image using Gemini's vision.
        
        Supports: JPEG, PNG, WebP, HEIC, HEIF formats
        This is the key differentiator vs regex/Tesseract approach.
        """
        if not self.is_available:
            return AIExtractedData(
                errors=["Gemini API not configured for image processing."],
                model_used="none",
            )

        image_part = {
            "mime_type": media_type,
            "data": image_bytes,
        }

        prompt = f"""This is a scanned Pakistani invoice/receipt image.
{RECEIPT_EXTRACTION_USER_TEMPLATE.format(invoice_text='[Image provided above - extract from the image]')}"""

        try:
            response = self.model.generate_content([image_part, prompt])
            raw = response.text
            groq_extractor = GroqReceiptExtractor.__new__(GroqReceiptExtractor)
            return groq_extractor._parse_response(raw, self.model_name + "-vision")

        except Exception as e:
            logger.error(f"Gemini image extraction failed: {e}")
            return AIExtractedData(
                errors=[f"Gemini vision extraction failed: {str(e)}"],
                model_used=self.model_name,
            )


# ─── AI Router (Primary + Fallback) ──────────────────────────────────────────

class SmartCFOAIRouter:
    """
    Intelligent routing layer that tries Groq first, falls back to Gemini,
    and finally falls back to regex extraction.
    
    This is the main interface used by the Streamlit app.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.groq = GroqReceiptExtractor(api_key=groq_api_key)
        self.gemini = GeminiReceiptExtractor(api_key=gemini_api_key)
        self._usage_stats = {"groq": 0, "gemini": 0, "fallback": 0, "errors": 0}

    @property
    def available_providers(self) -> List[str]:
        providers = []
        if self.groq.is_available:
            providers.append("groq")
        if self.gemini.is_available:
            providers.append("gemini")
        if not providers:
            providers.append("regex_fallback")
        return providers

    def extract_receipt(self, invoice_text: str) -> Tuple[AIExtractedData, str]:
        """
        Routes receipt extraction to the best available provider.
        
        Returns:
            Tuple of (AIExtractedData, provider_used)
        """
        # Try Groq first (fastest)
        if self.groq.is_available:
            result = self.groq.extract_receipt(invoice_text)
            if result.is_valid:
                self._usage_stats["groq"] += 1
                return result, "groq"
            else:
                logger.warning(f"Groq extraction invalid: {result.errors}. Trying Gemini...")

        # Try Gemini as fallback
        if self.gemini.is_available:
            result = self.gemini.extract_receipt(invoice_text)
            if result.is_valid:
                self._usage_stats["gemini"] += 1
                return result, "gemini"

        # Final fallback: regex extraction
        self._usage_stats["fallback"] += 1
        logger.warning("No AI providers available. Using regex fallback.")
        from utils.ocr_pipeline import extract_from_text
        regex_result = extract_from_text(invoice_text, source_hint="regex_fallback")
        fallback = AIExtractedData(
            vendor_name=regex_result.vendor_raw,
            amount_total=regex_result.amount_total,
            amount_tax=regex_result.amount_tax,
            invoice_number=regex_result.invoice_number,
            invoice_date=regex_result.invoice_date,
            expense_category=regex_result.category,
            extraction_confidence=regex_result.confidence,
            model_used="regex_fallback",
            errors=regex_result.errors,
        )
        return fallback, "regex_fallback"

    def extract_batch(
        self, receipts: list, delay_between: float = 0.5
    ) -> List[Tuple[AIExtractedData, str]]:
        """
        Batch processes multiple receipts with rate limit management.
        
        Args:
            receipts:       List of Receipt objects
            delay_between:  Seconds between API calls (avoid rate limits)
        
        Returns:
            List of (AIExtractedData, provider) tuples
        """
        results = []
        for i, receipt in enumerate(receipts):
            if i > 0:
                time.sleep(delay_between)
            result, provider = self.extract_receipt(receipt.raw_text)
            results.append((result, provider))
        return results

    def generate_cfo_insights(self, pl_data: dict) -> CFOInsight:
        """Generates CFO insights via the best available AI provider."""
        if self.groq.is_available:
            return self.groq.generate_cfo_insights(pl_data)
        return CFOInsight(
            summary="Configure GROQ_API_KEY or GEMINI_API_KEY to enable AI-powered CFO insights.",
            model_used="none",
        )

    @property
    def usage_stats(self) -> dict:
        return self._usage_stats.copy()


# ─── Mock AI (for demo without API keys) ─────────────────────────────────────

def mock_ai_extract(invoice_text: str) -> AIExtractedData:
    """
    Simulates AI extraction for demo purposes when no API key is configured.
    Uses regex extraction but formats it as an AIExtractedData response.
    This allows the UI to work completely without API keys.
    """
    from utils.ocr_pipeline import extract_from_text
    regex_result = extract_from_text(invoice_text, source_hint="demo_mock")

    return AIExtractedData(
        vendor_name=regex_result.vendor_raw or "Extracted Vendor",
        vendor_normalized=regex_result.vendor_normalized or regex_result.vendor_raw,
        invoice_number=regex_result.invoice_number,
        invoice_date=regex_result.invoice_date,
        amount_total=regex_result.amount_total,
        amount_subtotal=regex_result.amount_subtotal,
        amount_tax=regex_result.amount_tax,
        tax_rate_applied=18.0 if regex_result.amount_tax > 0 else 0.0,
        currency="PKR",
        expense_category=regex_result.category,
        is_recurring=regex_result.category in ["utilities", "rent", "internet_telecom", "salaries"],
        extraction_confidence=regex_result.confidence,
        model_used="demo_mode (configure API keys for AI)",
        ai_notes="Running in demo mode. Add GROQ_API_KEY or GEMINI_API_KEY for AI-powered extraction.",
        errors=regex_result.errors,
    )


def mock_cfo_insights(pl_data: dict) -> CFOInsight:
    """
    Generates rule-based CFO insights for demo mode (no API key required).
    Mirrors what an AI would generate using threshold-based logic.
    """
    net_margin = pl_data.get("net_margin", 0)
    gross_margin = pl_data.get("gross_margin", 0)
    revenue_growth = pl_data.get("revenue_growth_mom", 0)
    period = pl_data.get("period", "this period")

    summary_parts = []
    if revenue_growth > 10:
        summary_parts.append(f"Strong revenue growth of {revenue_growth:.1f}% MoM signals positive business momentum.")
    elif revenue_growth < -5:
        summary_parts.append(f"Revenue declined {abs(revenue_growth):.1f}% MoM — requires immediate attention.")
    else:
        summary_parts.append(f"Revenue remained relatively stable this period.")

    if net_margin > 10:
        summary_parts.append(f"Net margin of {net_margin:.1f}% reflects healthy operational efficiency.")
    elif net_margin < 0:
        summary_parts.append(f"Net loss of {abs(net_margin):.1f}% margin indicates critical cost management issues.")
    else:
        summary_parts.append(f"Net margin of {net_margin:.1f}% is below target — focus on cost optimization.")

    risks = []
    if gross_margin < 45:
        risks.append("COGS pressure eroding gross margins — renegotiate supplier contracts")
    if pl_data.get("expense_ratios", {}).get("salaries", 0) > 28:
        risks.append("Payroll costs elevated — review headcount productivity metrics")
    if pl_data.get("expense_ratios", {}).get("utilities", 0) > 4:
        risks.append("K-Electric costs high — evaluate solar investment ROI for Karachi operations")
    if not risks:
        risks.append("Monitor PKR/USD exchange rate impact on software subscription costs")
        risks.append("Maintain 3-month operating expense reserve for business continuity")

    tips = [
        "Negotiate early payment discounts with suppliers (2/10 net 30 terms)",
        "Review all SaaS subscriptions — consolidate overlapping tools",
        "Consider FBR tax credit utilization for technology investments under Section 65B",
    ]
    if pl_data.get("expense_ratios", {}).get("marketing", 0) < 3:
        tips.append("Marketing spend below 3% of revenue — digital marketing ROI in Pakistan is strong")

    return CFOInsight(
        summary=" ".join(summary_parts),
        key_risks=risks[:3],
        cost_optimization_tips=tips[:3],
        revenue_observations=f"{'Growing' if revenue_growth > 0 else 'Declining'} revenue trend. Target 15% MoM growth to meet annual targets. Focus on expanding recurring revenue streams.",
        cashflow_outlook="Maintain minimum 60-day cash reserves. Review receivables aging — Pakistani B2B payment cycles average 45-90 days.",
        tax_notes=f"Q4 advance tax payment due under Income Tax Ordinance 2001. Ensure NTN withholding certificates (Section 153) from clients are collected. STRN compliance for sales tax returns (monthly).",
        model_used="rule_based_demo",
    )


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.dummy_data import generate_monthly_receipts

    print("🤖 Testing SmartCFO AI Integration...")
    print("=" * 60)

    receipts = generate_monthly_receipts("Jan", 2024)
    test_receipt = receipts[0]  # First receipt

    print(f"📄 Testing with receipt: {test_receipt.vendor} ({test_receipt.category})")
    print(f"   Ground truth amount: PKR {test_receipt.amount:,.2f}\n")

    # Test mock extraction (works without API keys)
    result = mock_ai_extract(test_receipt.raw_text)
    print("🔍 Mock AI Extraction Result:")
    print(f"   Vendor    : {result.vendor_name}")
    print(f"   Amount    : PKR {result.amount_total:,.2f}")
    print(f"   Category  : {result.expense_category}")
    print(f"   Confidence: {result.extraction_confidence:.0%}")
    print(f"   Model     : {result.model_used}")

    # Check if API keys available
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    router = SmartCFOAIRouter(groq_api_key=groq_key, gemini_api_key=gemini_key)
    print(f"\n🔌 Available AI providers: {router.available_providers}")

    if groq_key or gemini_key:
        print("\n🚀 Testing real AI extraction...")
        ai_result, provider = router.extract_receipt(test_receipt.raw_text)
        print(f"   Provider  : {provider}")
        print(f"   Vendor    : {ai_result.vendor_name}")
        print(f"   Amount    : PKR {ai_result.amount_total:,.2f}")
        print(f"   Confidence: {ai_result.extraction_confidence:.0%}")
        if ai_result.ai_notes:
            print(f"   AI Notes  : {ai_result.ai_notes}")
    else:
        print("\n💡 Set GROQ_API_KEY or GEMINI_API_KEY for AI-powered extraction.")

    # Test mock CFO insights
    test_pl_data = {
        "company_name": "TechBridge Solutions",
        "period": "January 2024",
        "revenue": 2_850_000,
        "cogs": 798_000,
        "gross_profit": 2_052_000,
        "gross_margin": 72.0,
        "opex_total": 1_425_000,
        "ebit": 627_000,
        "ebit_margin": 22.0,
        "tax_provision": 181_830,
        "net_profit": 445_170,
        "net_margin": 15.6,
        "revenue_growth_mom": 8.5,
        "opex_breakdown": {
            "salaries": 627_000,
            "rent": 171_000,
            "utilities": 85_500,
            "internet_telecom": 57_000,
            "marketing": 114_000,
            "software": 57_000,
            "supplies": 42_750,
            "transport": 28_500,
            "professional_fees": 14_250,
        },
        "expense_ratios": {
            "salaries": 22.0, "rent": 6.0, "utilities": 3.0,
            "cogs": 28.0, "marketing": 4.0,
        },
    }

    insights = mock_cfo_insights(test_pl_data)
    print("\n💡 CFO Insights (Demo Mode):")
    print(f"   Summary: {insights.summary[:120]}...")
    print(f"   Risks  : {insights.key_risks[0] if insights.key_risks else 'N/A'}")
    print(f"   Tax    : {insights.tax_notes[:100]}...")
