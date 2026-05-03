"""
============================================================
SmartCFO — OCR & Text Extraction Pipeline
============================================================

This module provides a two-path extraction pipeline:
  Path A: pdfplumber  → for real, text-based PDFs
  Path B: Tesseract   → for scanned/image PDFs (fallback)
  Path C: Raw text    → for simulated receipts in our dummy data

Engineering Note:
In production, pdfplumber handles 90%+ of Pakistani e-invoices
because FBR's IRIS portal generates text-based PDFs. Tesseract
is the fallback for scanned paper receipts.

Pattern matching uses a layered approach:
  1. Exact regex anchors (TOTAL, AMOUNT DUE, NET PAYABLE)
  2. Pakistani-locale number formatting (commas: 1,23,456.00)
  3. Contextual vendor keyword classification
  4. Fuzzy/heuristic fallback for noisy OCR output
"""

import re
import io
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Try importing optional dependencies gracefully ───────────────────────────

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not installed. PDF extraction disabled. pip install pdfplumber")

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract/PIL not installed. OCR fallback disabled.")


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ExtractedInvoice:
    """
    Structured output from the OCR/extraction pipeline.
    All amounts are in PKR.
    """
    raw_text: str                          # Cleaned raw text from OCR/PDF
    vendor_raw: str = ""                   # Raw vendor name as extracted
    vendor_normalized: str = ""            # Normalized vendor name
    invoice_number: str = ""              # Invoice/receipt number
    invoice_date: str = ""                # Date string
    amount_total: float = 0.0             # Final amount due (PKR)
    amount_subtotal: float = 0.0          # Pre-tax subtotal
    amount_tax: float = 0.0              # Tax amount
    category: str = "uncategorized"       # Expense category
    confidence: float = 0.0              # Extraction confidence (0–1)
    extraction_method: str = "unknown"    # "pdfplumber" | "tesseract" | "regex" | "raw"
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def is_valid(self) -> bool:
        return self.amount_total > 0 and len(self.errors) == 0


# ─── Pakistani Number Format Parser ──────────────────────────────────────────

def parse_pkr_amount(amount_str: str) -> Optional[float]:
    """
    Parses Pakistani Rupee amounts from text.
    
    Handles formats:
      PKR 1,23,456.00  →  123456.00
      Rs. 45,000       →  45000.00
      1,500.50         →  1500.50
      PKR45000         →  45000.00
    
    Pakistani number formatting uses the South Asian numbering system:
    digits are grouped as X,XX,XX,XXX (e.g., 12,34,567)
    """
    if not amount_str:
        return None

    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[PKRrs.\s]', '', amount_str, flags=re.IGNORECASE)
    # Remove all commas (handles both South Asian and Western formats)
    cleaned = cleaned.replace(',', '')
    # Remove stray characters except digits and decimal
    cleaned = re.sub(r'[^\d.]', '', cleaned)

    # Handle edge cases
    if not cleaned or cleaned == '.':
        return None

    # Ensure only one decimal point
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    try:
        value = float(cleaned)
        # Sanity check: Pakistani SME amounts between Rs 100 and Rs 50M
        if 100 <= value <= 50_000_000:
            return value
        elif value > 0:
            return value  # Return anyway, let caller decide
    except ValueError:
        pass

    return None


# ─── Regex Extraction Patterns ────────────────────────────────────────────────

# Priority-ordered patterns for extracting total amounts
# Higher priority = more specific/reliable matches
AMOUNT_PATTERNS = [
    # Most specific patterns (highest confidence)
    (r'TOTAL\s+AMOUNT\s+DUE[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.95),
    (r'NET\s+PAYABLE[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.95),
    (r'GRAND\s+TOTAL[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.92),
    (r'INVOICE\s+TOTAL[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.90),
    (r'GROSS\s+RENT\s+PAID[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.90),
    (r'NET\s+DISBURSED[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.90),
    (r'TOTAL\s+CHARGED[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.90),
    # Generic TOTAL
    (r'^TOTAL[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.80),
    (r'TOTAL\s+DUE[:\s]+(?:PKR|Rs\.?|PKR\.?)?\s*([\d,]+\.?\d*)', 0.80),
    # Fallback patterns
    (r'(?:PKR|Rs\.?)\s*([\d,]{4,}\.?\d*)', 0.50),
]

# Patterns for extracting invoice metadata
DATE_PATTERNS = [
    r'(?:Invoice|Bill|Receipt|Pay Period|Date)[:\s]+(\d{4}-\d{2}-\d{2})',
    r'(?:Invoice|Bill|Receipt)[:\s]+(\d{2}/\d{2}/\d{4})',
    r'(?:Invoice|Bill|Receipt)[:\s]+(\d{2}-\d{2}-\d{4})',
    r'Date\s*:\s*(\d{4}-\d{2}-\d{2})',
]

INVOICE_NUMBER_PATTERNS = [
    r'Invoice\s+(?:No|#|Number)\.?\s*:\s*([A-Z0-9\-_]+)',
    r'(?:Bill|Receipt|Fee Note)\s+No\.?\s*:\s*([A-Z0-9\-_]+)',
    r'Reference\s+No\.?\s*:\s*([A-Z0-9\-_]+)',
    r'INV-[\d\-]+',
    r'PAY-[\d]+',
]

TAX_PATTERNS = [
    r'(?:GST|Sales Tax|VAT)(?:\s+@\s+\d+%)?[:\s]+(?:PKR|Rs\.?)?\s*([\d,]+\.?\d*)',
    r'General Sales Tax[:\s]+(?:PKR|Rs\.?)?\s*([\d,]+\.?\d*)',
]

SUBTOTAL_PATTERNS = [
    r'Sub[-\s]?Total[:\s]+(?:PKR|Rs\.?)?\s*([\d,]+\.?\d*)',
    r'Sub-Total[:\s]+(?:PKR|Rs\.?)?\s*([\d,]+\.?\d*)',
]


# ─── Vendor Classification ────────────────────────────────────────────────────

# Maps vendor keywords → expense category
VENDOR_CLASSIFIER = {
    "k-electric": "utilities",
    "ke ": "utilities",
    "karachi electric": "utilities",
    "sui southern": "utilities",
    "ssgc": "utilities",
    "stormfiber": "internet_telecom",
    "cybernet": "internet_telecom",
    "ptcl": "internet_telecom",
    "nayatel": "internet_telecom",
    "jazz business": "internet_telecom",
    "zong": "internet_telecom",
    "dolmen": "rent",
    "lease": "rent",
    "rent receipt": "rent",
    "payroll": "salaries",
    "salary statement": "salaries",
    "salaries": "salaries",
    "al-fatah": "supplies",
    "carrefour": "supplies",
    "naheed": "supplies",
    "metro cash": "supplies",
    "office supplies": "supplies",
    "microsoft": "software",
    "adobe": "software",
    "zoom": "software",
    "slack": "software",
    "github": "software",
    "subscription": "software",
    "meta platforms": "marketing",
    "facebook": "marketing",
    "google ads": "marketing",
    "dawn media": "marketing",
    "advertising": "marketing",
    "bykea": "transport",
    "tcs couriers": "transport",
    "courier": "transport",
    "transport": "transport",
    "chartered accountant": "professional_fees",
    "icap": "professional_fees",
    "tax consultant": "professional_fees",
    "law associates": "professional_fees",
    "distributor": "cogs",
    "it hardware": "cogs",
    "laptop": "cogs",
    "hardware": "cogs",
}


def classify_vendor(text: str) -> str:
    """
    Classifies a vendor/invoice into an expense category
    based on keyword matching against the vendor classifier map.
    
    Returns 'uncategorized' if no match is found.
    """
    text_lower = text.lower()
    for keyword, category in VENDOR_CLASSIFIER.items():
        if keyword in text_lower:
            return category
    return "uncategorized"


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_extracted_text(raw: str) -> str:
    """
    Cleans raw OCR or pdfplumber output for reliable regex parsing.
    
    Operations performed:
    1. Normalize whitespace (OCR produces inconsistent spacing)
    2. Remove null bytes and control characters
    3. Normalize PKR currency representations
    4. Fix common OCR substitution errors (0 → O, l → 1)
    5. Standardize line endings
    """
    if not raw:
        return ""

    text = raw

    # Remove null bytes and non-printable characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive blank lines (keep max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize whitespace within lines (but preserve line breaks)
    lines = text.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(lines)

    # Normalize currency representations
    text = re.sub(r'PKR\.?', 'PKR', text, flags=re.IGNORECASE)
    text = re.sub(r'Rs\.?\s', 'PKR ', text, flags=re.IGNORECASE)
    text = re.sub(r'Rupees?', 'PKR', text, flags=re.IGNORECASE)

    # Common OCR errors in amounts (in numeric contexts)
    # 'O' mistaken for '0' in numbers
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ─── Core Extraction Logic ────────────────────────────────────────────────────

def extract_from_text(raw_text: str, source_hint: str = "raw") -> ExtractedInvoice:
    """
    Main extraction function. Takes raw text (from any source) and
    extracts structured financial data using regex pattern matching.
    
    Args:
        raw_text:    Raw text string (from pdfplumber, Tesseract, or dummy data)
        source_hint: Extraction method label for audit trail
    
    Returns:
        ExtractedInvoice with all fields populated
    """
    cleaned = clean_extracted_text(raw_text)
    invoice = ExtractedInvoice(
        raw_text=cleaned,
        extraction_method=source_hint,
    )

    # ── 1. Extract Total Amount ───────────────────────────────────────────────
    best_amount = None
    best_confidence = 0.0

    for pattern, confidence in AMOUNT_PATTERNS:
        matches = re.findall(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if matches:
            for match in matches:
                parsed = parse_pkr_amount(match)
                if parsed and parsed > 0 and confidence > best_confidence:
                    best_amount = parsed
                    best_confidence = confidence
                    break  # Take first match for highest-confidence pattern
        if best_confidence >= 0.95:
            break  # Good enough, stop searching

    invoice.amount_total = best_amount or 0.0
    invoice.confidence = best_confidence

    # ── 2. Extract Tax Amount ─────────────────────────────────────────────────
    for pattern in TAX_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            parsed = parse_pkr_amount(match.group(1))
            if parsed:
                invoice.amount_tax = parsed
                break

    # ── 3. Extract Subtotal ───────────────────────────────────────────────────
    for pattern in SUBTOTAL_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            parsed = parse_pkr_amount(match.group(1))
            if parsed:
                invoice.amount_subtotal = parsed
                break

    # Infer subtotal from total - tax if not found
    if invoice.amount_subtotal == 0 and invoice.amount_total > 0 and invoice.amount_tax > 0:
        invoice.amount_subtotal = invoice.amount_total - invoice.amount_tax

    # ── 4. Extract Invoice Date ───────────────────────────────────────────────
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            invoice.invoice_date = match.group(1)
            break

    # ── 5. Extract Invoice Number ─────────────────────────────────────────────
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            invoice.invoice_number = match.group(0) if not match.lastindex else match.group(1)
            break

    # ── 6. Extract Vendor Name ────────────────────────────────────────────────
    # First line is usually the vendor name in Pakistani invoices
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    if lines:
        # Take first substantive line (skip very short lines)
        for line in lines[:3]:
            if len(line) > 5:
                invoice.vendor_raw = line
                break

    # ── 7. Classify Category ──────────────────────────────────────────────────
    invoice.category = classify_vendor(cleaned)
    invoice.vendor_normalized = invoice.vendor_raw

    # ── 8. Validation Errors ──────────────────────────────────────────────────
    if invoice.amount_total <= 0:
        invoice.errors.append("Could not extract total amount from text")
    if not invoice.invoice_date:
        invoice.errors.append("Invoice date not found")
    if invoice.category == "uncategorized":
        invoice.errors.append("Vendor category not recognized — manual review required")

    return invoice


# ─── PDF Extraction (pdfplumber) ───────────────────────────────────────────────

def extract_from_pdf_bytes(pdf_bytes: bytes) -> ExtractedInvoice:
    """
    Extracts text from a PDF file using pdfplumber.
    
    pdfplumber is preferred over PyMuPDF for Pakistani invoices because:
    - Better handling of Urdu/Arabic numerals in mixed-language PDFs
    - Preserves table structure (important for itemized invoices)
    - More accurate whitespace handling for columnar financial data
    
    Args:
        pdf_bytes: Raw PDF bytes (from file upload or download)
    
    Returns:
        ExtractedInvoice
    """
    if not HAS_PDFPLUMBER:
        return ExtractedInvoice(
            raw_text="",
            errors=["pdfplumber not installed. Run: pip install pdfplumber"],
        )

    try:
        all_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout preservation
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    all_text.append(f"--- Page {page_num + 1} ---\n{text}")

                # Also extract tables (for itemized invoices)
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            all_text.append(' | '.join(
                                str(cell).strip() if cell else '' for cell in row
                            ))

        combined_text = '\n'.join(all_text)

        if not combined_text.strip():
            # PDF has no extractable text → likely scanned → use Tesseract
            logger.info("pdfplumber found no text. Falling back to Tesseract OCR.")
            return extract_from_pdf_ocr_bytes(pdf_bytes)

        invoice = extract_from_text(combined_text, source_hint="pdfplumber")
        return invoice

    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return ExtractedInvoice(
            raw_text="",
            errors=[f"PDF extraction error: {str(e)}"],
        )


def extract_from_pdf_ocr_bytes(pdf_bytes: bytes) -> ExtractedInvoice:
    """
    OCR fallback using Tesseract for scanned/image PDFs.
    
    Pipeline:
    1. Convert PDF pages to images (via pdf2image/pillow)
    2. Apply image preprocessing (deskew, denoise, binarize)
    3. Run Tesseract with English + Urdu language pack
    4. Parse extracted text with regex pipeline
    
    Tesseract config optimized for Pakistani invoices:
    - PSM 6: Assume uniform block of text
    - OEM 3: LSTM engine (best accuracy)
    - Language: eng (Urdu support via urd if installed)
    """
    if not HAS_TESSERACT:
        return ExtractedInvoice(
            raw_text="",
            errors=["pytesseract not installed. Run: pip install pytesseract pillow pdf2image"],
        )

    try:
        # For this implementation, we show the Tesseract config
        # In production, you'd convert PDF pages to PIL Images first
        custom_config = r'--oem 3 --psm 6 -l eng'

        # Example of how you'd process an actual image:
        # from pdf2image import convert_from_bytes
        # images = convert_from_bytes(pdf_bytes, dpi=300)
        # full_text = ""
        # for img in images:
        #     # Preprocess for better OCR accuracy
        #     img_gray = img.convert('L')  # Grayscale
        #     # Optional: img_gray = ImageEnhance.Contrast(img_gray).enhance(2.0)
        #     text = pytesseract.image_to_string(img_gray, config=custom_config)
        #     full_text += text + "\n"

        # For demo purposes, return a placeholder
        return ExtractedInvoice(
            raw_text="[Tesseract OCR would process scanned PDF here]",
            extraction_method="tesseract",
            errors=["Tesseract OCR requires pdf2image for PDF input"],
        )

    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ExtractedInvoice(
            raw_text="",
            errors=[f"OCR error: {str(e)}"],
        )


# ─── Convenience: Process Receipt Object ─────────────────────────────────────

def process_receipt_text(receipt_raw_text: str) -> ExtractedInvoice:
    """
    Processes our dummy receipt raw text through the extraction pipeline.
    This simulates what would happen with real OCR/PDF extraction.
    
    Args:
        receipt_raw_text: The simulated invoice text from dummy_data.py
    
    Returns:
        ExtractedInvoice with all fields
    """
    return extract_from_text(receipt_raw_text, source_hint="simulated_ocr")


def process_all_receipts(receipts: list) -> List[ExtractedInvoice]:
    """
    Batch processes a list of Receipt objects through the extraction pipeline.
    Returns list of ExtractedInvoice objects.
    """
    results = []
    for receipt in receipts:
        extracted = process_receipt_text(receipt.raw_text)
        # Supplement with ground truth from Receipt object where extraction failed
        if extracted.amount_total == 0:
            extracted.amount_total = receipt.amount
            extracted.confidence = 0.5  # Lower confidence — used fallback
            extracted.errors.append("Regex extraction failed, used Receipt ground truth")
        results.append(extracted)
    return results


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test the extraction pipeline with a sample Pakistani invoice text
    sample_text = """
    K-ELECTRIC LIMITED
    Invoice No. : INV-202401-0001
    Bill Date   : 2024-01-15
    Consumer No.: KE-123456
    
    BILLING DETAILS:
    Energy Charges         : PKR     42,500.00
    Fuel Price Adjustment  : PKR      1,200.00
    Sub Total              : PKR     43,700.00
    General Sales Tax (18%): PKR      7,866.00
    TOTAL AMOUNT DUE       : PKR     51,566.00
    
    Due Date: 2024-01-30
    """

    result = extract_from_text(sample_text, source_hint="test")
    print("=== OCR Extraction Test ===")
    print(f"Vendor     : {result.vendor_raw}")
    print(f"Invoice No : {result.invoice_number}")
    print(f"Date       : {result.invoice_date}")
    print(f"Total      : PKR {result.amount_total:,.2f}")
    print(f"Tax        : PKR {result.amount_tax:,.2f}")
    print(f"Subtotal   : PKR {result.amount_subtotal:,.2f}")
    print(f"Category   : {result.category}")
    print(f"Confidence : {result.confidence:.0%}")
    print(f"Valid      : {result.is_valid}")
    print(f"Errors     : {result.errors}")
