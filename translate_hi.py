from __future__ import annotations
from transformers import pipeline
from typing import List, Dict

_TRANSLATOR = None

def build_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    return _TRANSLATOR

def translate_segments(segments: List[Dict], translator=None) -> List[Dict]:
    if translator is None:
        translator = build_translator()
    texts = [s["text"] for s in segments]
    if not texts:
        return []
    results = translator(texts, clean_up_tokenization_spaces=True)
    hi_texts = [r["translation_text"] for r in results]
    out = []
    for s, t in zip(segments, hi_texts):
        x = dict(s)
        x["text_hi"] = t
        out.append(x)
    return out
