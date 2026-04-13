import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)


class GroqCleaner:
    NON_PRODUCT_PATTERNS = [
        r"\bthank you\b", r"\bplease come again\b", r"\bgoods sold are not\b",
        r"\bprivilege card\b", r"\bdocument\b", r"\bdiscount\b",
        r"\bsubtotal\b", r"\btotal\b", r"\bcash\b", r"\bchange\b",
        r"\badjustment\b", r"\brounding adjustment\b", r"\bwholesale\b",
        r"\bcustomer service\b", r"\bhotline\b", r"\bfeedback\b",
        r"\bdownload\b", r"\bgoogle play\b", r"\bapp store\b", r"\blicensee\b",
    ]

    ADDRESS_PATTERNS = [
        r"\bjalan\b", r"\bjohor\b", r"\bselangor\b", r"\bkawasan\b",
        r"\bperindustrian\b", r"\bseri\b", r"\btaman\b", r"\bno\b",
        r"\blot\b", r"\blevel\b", r"\bbangunan\b", r"\btel\b", r"\b\d{5}\b",
    ]

    BUSINESS_PATTERNS = [
        r"\bsdn\b", r"\bbhd\b", r"\benterprise\b", r"\btrading\b",
    ]

    PRODUCT_KEYWORDS = {
        "tape", "lamp", "sprayer", "cleaner", "bag", "board", "atta",
        "oil", "handkerchief", "wash", "wax", "bopp", "windshield",
        "glass", "automotive", "plastic", "rubber", "electronic",
        "cable", "battery", "motor", "filter", "valve", "pipe",
        "steel", "aluminium", "machine", "tool",
    }

    BAD_OUTPUT_PATTERNS = [
        r"\bthere is no\b", r"\bplease provide\b", r"\bit seems like\b",
        r"\bit appears\b", r"\bnot a product\b", r"\baddress\b",
        r"\bbusiness name\b", r"\bgreeting\b", r"\bstatement\b",
        r"\bno output\b", r"\bcustomer service\b", r"\bfeedback\b",
        r"\bdownload\b", r"\bgoogle play\b", r"\bapp store\b",
    ]

    def __init__(self, model="llama-3.1-8b-instant"):
        self.model = model

    def clean(self, raw_text: str) -> str:
        normalized = self._normalize_text(raw_text)
        if not normalized or self._is_non_product_line(normalized):
            return ""

        prompt = f"""
You normalize noisy invoice lines for HS-code retrieval.

Task:
Convert the input into one short lowercase product description.

Rules:
- Output only the cleaned product text
- Do not add labels like "product description:"
- Do not explain your reasoning
- Remove prices, totals, invoice ids, tax ids, store messages, addresses, and company names
- Remove obvious brand or marketing words when they are not the product itself
- Keep the core merchandise meaning
- If the text is not a product line, output exactly: skip

Input:
{normalized}
"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only a short cleaned product phrase or skip.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=40,
            )
            if (
                response is None
                or not getattr(response, "choices", None)
                or not response.choices[0].message.content
            ):
                raise ValueError("Groq returned no text")

            cleaned = self._postprocess_output(response.choices[0].message.content)
            if cleaned:
                return cleaned
        except Exception:
            pass

        return self._local_clean(normalized)

    def clean_lines(self, raw_invoice: str) -> list:
        lines = raw_invoice.splitlines()
        products = [line.strip() for line in lines if line.strip()]
        cleaned_lines = []
        buffer = ""

        for product in products:
            cleaned = self.clean(product)
            if not cleaned:
                continue
            if len(cleaned.split()) <= 2:
                buffer = f"{buffer} {cleaned}".strip()
                continue
            if buffer:
                cleaned = f"{buffer} {cleaned}"
                buffer = ""
            cleaned_lines.append({
                "text": cleaned,
                "confidence": self.confidence(cleaned)
            })

        if buffer:
            cleaned_lines.append(buffer)
        return cleaned_lines

    def confidence(self, text):
        tokens = text.split()
        if len(tokens) >= 3:
            return 0.9
        elif len(tokens) == 2:
            return 0.7
        elif len(tokens) == 1:
            return 0.5
        return 0.3

    def _normalize_text(self, text: str) -> str:
        text = str(text or "").strip()
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_non_product_line(self, text: str) -> bool:
        lower = text.lower()
        if not re.search(r"[a-z]", lower):
            return True
        if any(re.search(pattern, lower) for pattern in self.NON_PRODUCT_PATTERNS):
            return True

        address_hits = sum(bool(re.search(p, lower)) for p in self.ADDRESS_PATTERNS)
        business_hits = sum(bool(re.search(p, lower)) for p in self.BUSINESS_PATTERNS)
        product_hits = sum(keyword in lower for keyword in self.PRODUCT_KEYWORDS)

        if lower.startswith("(") and lower.endswith(")") and product_hits == 0:
            return True
        if address_hits >= 2 and product_hits == 0:
            return True
        if business_hits >= 1 and product_hits == 0:
            return True
        if len(lower.split()) >= 3 and product_hits == 0 and address_hits >= 1:
            return True

        return False

    def _postprocess_output(self, text: str) -> str:
        cleaned = str(text or "").strip().lower()
        cleaned = cleaned.replace('"', "").replace("'", "")
        cleaned = re.sub(r"^product description\s*:\s*", "", cleaned)
        cleaned = re.sub(r"^cleaned product text\s*:\s*", "", cleaned)
        cleaned = cleaned.splitlines()[0].strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)

        if cleaned == "skip":
            return ""
        if any(re.search(pattern, cleaned) for pattern in self.BAD_OUTPUT_PATTERNS):
            return ""

        return self._local_clean(cleaned)

    def _local_clean(self, text: str) -> str:
        lower = text.lower()
        if self._is_non_product_line(lower):
            return ""

        lower = re.sub(r"\bwindshiled\b", "windshield", lower)
        lower = re.sub(r"\bboaro\b", "board", lower)
        lower = re.sub(
            r"\b(?:gstin|invoice|batch|hsn|qty|quantity|subtotal|total|amount)\b",
            " ", lower
        )
        lower = re.sub(r"[^a-z0-9#\-/\. ]", " ", lower)
        lower = re.sub(r"[/#]+", " ", lower)

        stopwords = {
            "mrp", "gst", "tax", "card", "document", "discount", "thank",
            "you", "please", "again", "customer", "service", "hotline",
            "feedback", "download", "google", "play", "app", "store",
            "licensee", "goods", "sold", "return", "returnap", "retur",
            "bangunan", "level", "tel", "johor", "selangor", "jalan",
            "kawasan", "perindustrian", "seri", "sdn", "bhd",
        }

        tokens = []
        for token in lower.split():
            if token in stopwords:
                continue
            token = token.strip("-.")
            if not token:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", token):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?(?:mm|cm|m|ml|l|kg|g|pcs)", token):
                tokens.append(token)
                continue
            if re.search(r"[a-z]", token):
                if len(token) == 1:
                    continue
                tokens.append(token)

        cleaned = " ".join(tokens)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if self._is_non_product_line(cleaned):
            return ""

        return cleaned