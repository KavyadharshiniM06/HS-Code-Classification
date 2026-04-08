import re
from typing import List, Dict
from collections import Counter

class HSCodeGenerator:
    def __init__(
        self,
        llm,
        max_tokens=128, # Reduced for speed/focus
        temperature=0.1 # Lower temperature for higher consistency
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _estimate_retrieval_confidence(self, context):
        scores = re.findall(r"SCORE:\s*([0-9.]+)", context)

        if not scores:
            return 0.5

        scores = [float(s) for s in scores]
        return sum(scores[:3]) / min(len(scores), 3)

    def generate(self, query: str, augmented_context: str) -> Dict:
        if not augmented_context.strip():
            return {
                "prediction": None,
                "confidence": 0.0,
                "reasoning": "No supporting evidence retrieved."
            }

        prompt = self._build_prompt(query, augmented_context)
        
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        parsed = self._parse_response(response)

        valid_codes = re.findall(r'HS CODE:\s*(\d{6})', augmented_context)

        if parsed["prediction"] not in valid_codes:
            parsed["confidence"] *= 0.5

        # Boost using retrieval confidence
        retrieval_conf = self._estimate_retrieval_confidence(augmented_context)

        if parsed["confidence"] > 0:
            parsed["confidence"] = round(
                (parsed["confidence"] * 0.7) + (retrieval_conf * 0.3),
                3
            )

        return parsed

    # --------------------------------------------------
    # SIMPLIFIED PROMPT (Local-Model Friendly)
    # --------------------------------------------------
    def _build_prompt(self, query, context):
        # We use a structured "Question/Answer" style which Flan-T5 was trained on.
        return f"""Task: Select the single best 6-digit HS code for the product using only the evidence candidates below.

Evidence:
{context}

Product: {query}

Instructions:
1. Choose only from the HS codes shown in the evidence.
2. Do not invent or add any HS code that is not listed.
3. If the product does not clearly match any evidence candidate, answer NONE.
4. Provide a confidence score between 0.0 and 1.0.

Answer Format:
HS_CODE: [6-digit code]
CONFIDENCE: [score]
REASONING: [explanation]

Answer:"""

    # --------------------------------------------------
    # ROBUST PARSING (Regex based)
    # --------------------------------------------------
    def _parse_response(self, text):
        # Default state
        result = {
            "prediction": None,
            "confidence": 0.0,
            "reasoning": text.strip() # Keep raw text as reasoning if parsing fails
        }

        # 1. Extract 6-digit HS Code using Regex
        # Looks for any string of 4 to 6 digits (common in HS codes)
        matches = re.findall(r'\b(\d{6})\b', text)

        if matches:
            result["prediction"] =  Counter(matches).most_common(1)[0][0]
        
        # 2. Extract Confidence Score
        # Looks for a float/decimal between 0.0 and 1.0
        conf_match = re.search(r'\b(0\.\d+|1\.0|1)\b', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                result["confidence"] = 0.5

        # 3. Clean up reasoning
        # If the model used the label 'REASONING:', extract just that part.
        if "REASONING:" in text:
            result["reasoning"] = text.split("REASONING:")[-1].strip()

        # Final sanity check: if the model said "NONE", nullify the prediction
        if "NONE" in text.upper() and not hs_match:
            result["prediction"] = None
            result["confidence"] = 0.0

        return result