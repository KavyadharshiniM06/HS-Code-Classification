import re


class QueryReformulator:
    """
    Reformulates noisy parsed text into
    retrieval-optimized query strings for HS code search.
    """

    def __init__(self):

        # Measurement / quantity units
        self.units = {
            "L", "LTR", "LITRE", "LITER",
            "ML", "CL", "DL",
            "KG", "KGS", "G", "GM", "GRAM", "GRAMS",
            "MM", "CM", "M", "MTR", "METER", "METRE",
            "IN", "INCH", "INCHES",
            "FT", "FEET",
            "PCS", "PC", "NOS", "NO", "QTY",
            "SET", "PAIR",
            "BOX", "CARTON", "CTN",
            "PACK", "PK"
        }

        # Invoice / commercial noise
        self.invoice_noise = {
            "GST", "CGST", "SGST", "IGST",
            "VAT", "TAX",
            "TOTAL", "SUBTOTAL",
            "AMOUNT", "NET", "GROSS",
            "INVOICE", "RECEIPT",
            "BILL", "DATE", "TIME",
            "CASH", "CARD",
            "MRP", "PRICE", "RATE",
            "DISCOUNT", "DISC"
        }

        # Style / catalog noise
        self.style_noise = {
            "MODEL", "TYPE", "SERIES",
            "STYLE", "DESIGN",
            "ITEM", "CODE", "REF", "REFNO",
            "BATCH", "LOT",
            "VERSION", "VARIANT",
            "STITCH", "PREMIUM",
            "DELUXE", "SUPER",
            "PLUS", "PRO", "NEW"
        }

    def rewrite(self, text: str) -> str:
        text = text.upper()

        # Remove standalone numbers (preserve 12V, 3PLY, etc.)
        text = re.sub(r"\b\d+\b", "", text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()
        cleaned_tokens = []

        for token in tokens:

            # Remove single-letter tokens (I, A, B etc.)
            if len(token) <= 1:
                continue

            # Remove noise categories
            if (
                token in self.units
                or token in self.invoice_noise
                or token in self.style_noise
            ):
                continue

            # Remove pure numeric leftovers
            if token.isdigit():
                continue

            # Remove OCR garbage (mostly non-letters)
            alpha_ratio = sum(c.isalpha() for c in token) / max(len(token), 1)
            if alpha_ratio < 0.5:
                continue

            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens).strip()