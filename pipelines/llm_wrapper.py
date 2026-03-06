from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalLLM:
    def __init__(
        self,
        model_name="google/flan-t5-base",
        max_new_tokens=64
    ):
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = None,temperature:float=0.0):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
