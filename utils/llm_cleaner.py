import re
from google import genai
import os
API_KEY=os.getenv("API_GEMINI1")
client = genai.Client(api_key=API_KEY)

class GeminiCleaner:
    def __init__(self, model="models/gemini-2.5-flash"):
        self.model = model

    def clean(self, raw_text: str) -> str:
        """
        Try to clean with Gemini API; fallback to local cleaning if API fails
        """
        prompt = f"""
You are a customs trade text normalization assistant.

Task:
Convert raw invoice or receipt product text into a short, standardized trade description
that can be used for HS code retrieval.

Rules:
- Remove prices, quantities, invoice numbers, GST/VAT details, totals,address of the shop
- Remove brand names and marketing words
- Preserve the core product meaning
- Use neutral trade terminology (not HS codes)
- Do NOT infer or guess the HS chapter or code
- Do NOT add new product functionality
- Output ONE concise line only
- Use lowercase

Product text:
{raw_text}

Output:
"""
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.2, "max_output_tokens": 60}
            )
            # check if response is None or empty
            if response is None or not hasattr(response, "text") or not response.text:
                raise ValueError("Gemini returned no text")

            return response.text.strip().lower()
        except Exception:
            # fallback: simple regex-based cleaning
            text = re.sub(r"[\d\.,₹$]", "", raw_text)
            text = re.sub(r"MRP|GSTIN|TOTAL|Invoice|Batch|HSN", "", text, flags=re.I)
            return text.lower().strip()

    def clean_lines(self, raw_invoice: str) -> list:
        """
        Splits the invoice into lines and cleans each line individually.
        Returns a list of cleaned product descriptions.
        """
        lines = raw_invoice.splitlines()
        products = [line.strip() for line in lines if line.strip()]
        cleaned_lines = []
        for product in products:
            cleaned = self.clean(product)
            if cleaned:
                cleaned_lines.append(cleaned)
        return cleaned_lines


