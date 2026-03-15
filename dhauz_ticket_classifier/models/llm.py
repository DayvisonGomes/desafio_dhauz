"""LLM pipeline wrapper (HuggingFace pipeline)"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import torch


class LLMPipeline:
    def __init__(self, model_name: str = None, device_map='auto', dtype=torch.float16):
        self.model_name = model_name
        if model_name is None:
            raise ValueError("model_name is required")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)
        gen_config = GenerationConfig(do_sample=False, max_new_tokens=64, repetition_penalty=1.1)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            config=gen_config,
            return_full_text=False,
        )

    def generate(self, prompts, batch_size=1):
        return self.pipe(prompts, batch_size=batch_size, return_full_text=False)

    def cleanup(self):
        torch.cuda.empty_cache()
