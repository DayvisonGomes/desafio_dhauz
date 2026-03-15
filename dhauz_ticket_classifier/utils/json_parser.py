"""Helpers to extract JSON from LLM outputs"""
import json
import re
from typing import Optional, Dict

def extract_json(text: str) -> Optional[Dict]:
    matches = re.findall(r'\{[^{}]*\}', text)
    for m in matches:
        try:
            data = json.loads(m)
            if "class" in data and "justification" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None
