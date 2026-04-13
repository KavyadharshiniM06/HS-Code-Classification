import re


class QueryReformulator:
    """
    Reformulates noisy parsed text into
    retrieval-optimized query strings for HS code search.
    """

    def __init__(self):
        self.units = {
            "L", "LTR", "LITRE", "LITER", "ML", "CL", "DL",
            "KG", "KGS", "G", "GM", "GRAM", "GRAMS",
            "MM", "CM", "M", "MTR", "METER", "METRE",
            "IN", "INCH", "INCHES", "FT", "FEET",
            "PCS", "PC", "NOS", "NO", "QTY",
            "SET", "PAIR", "BOX", "CARTON", "CTN", "PACK", "PK"
        }

        self.invoice_noise = {
            "GST", "CGST", "SGST", "IGST", "VAT", "TAX",
            "TOTAL", "SUBTOTAL", "AMOUNT", "NET", "GROSS",
            "INVOICE", "RECEIPT", "BILL", "DATE", "TIME",
            "CASH", "CARD", "MRP", "PRICE", "RATE", "DISCOUNT", "DISC"
        }

        self.style_noise = {
            "MODEL", "TYPE", "SERIES", "STYLE", "DESIGN",
            "ITEM", "CODE", "REF", "REFNO", "BATCH", "LOT",
            "VERSION", "VARIANT", "STITCH", "PREMIUM",
            "DELUXE", "SUPER", "PLUS", "PRO", "NEW"
        }

    def rewrite(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()
        cleaned_tokens = []

        for token in tokens:
            if len(token) <= 1:
                continue
            if (
                token in self.units
                or token in self.invoice_noise
                or token in self.style_noise
            ):
                continue
            if token.isdigit():
                continue
            alpha_ratio = sum(c.isalpha() for c in token) / max(len(token), 1)
            if alpha_ratio < 0.5:
                continue
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens).strip()