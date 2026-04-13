import re
from typing import List


class ReceiptParser:
    """
    Generic parser for noisy commercial text:
    receipts, invoices, OCR output, PDFs.
    """

    def __init__(self):
        self.noise_patterns = [
            r"GSTIN.*",
            r"TOTAL.*",
            r"INVOICE.*",
            r"MRP.*",
            r"CGST.*",
            r"SGST.*",
            r"IGST.*",
            r"AMOUNT.*",
            r"DATE.*",
            r"PHONE.*",
            r"\d{10,}"
        ]

    def parse(self, raw_text: str) -> List[str]:
        lines = raw_text.splitlines()
        cleaned = []
        buffer = ""

        for line in lines:
            line = line.strip().upper()
            if not line:
                continue

            if self._is_noise(line):
                continue

            if self._is_candidate_text(line):
                if len(line.split()) <= 2:
                    buffer = f"{buffer} {line}".strip()
                    continue

                if buffer:
                    line = f"{buffer} {line}"
                    buffer = ""

                cleaned.append(line)

        if buffer:
            cleaned.append(buffer)

        cleaned = list(dict.fromkeys(cleaned))
        return cleaned

    def is_non_product(self, text: str) -> bool:
        blacklist = [
            "cash", "change", "total", "rounding", "refund",
            "receipt", "invoice", "exchange", "days",
            "allowed", "rm", "gst", "date", "time", "item(s)", "cashier"
        ]
        t = text.lower()
        return any(k in t for k in blacklist)

    def _is_noise(self, text: str) -> bool:
        for pattern in self.noise_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _is_candidate_text(self, text: str) -> bool:
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        word_count = len(text.split())
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        has_keywords = any(
            k in text.lower() for k in [
                "ml", "l", "cm", "mm", "kg", "g",
                "board", "tape", "sprayer", "cleaner",
                "glass", "plastic", "metal",
                "cable", "wire", "battery",
                "oil", "filter", "valve",
                "machine", "tool", "kit"
            ]
        )

        if not has_letters or word_count < 2:
            return False

        return alpha_ratio > 0.35 or has_digits or has_keywords