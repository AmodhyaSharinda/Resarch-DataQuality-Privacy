# tools/translation_tool.py

from langchain.tools import tool
from agents.text.translation_agent import TranslationAgent
from agents.text.lang_detect_agent import LanguageDetectionAgent
import json

@tool("translator_agent", description="Translates text to the target language. Detects source language automatically if not provided.")
def translation_tool(text: str, source_lang: str = None, target_lang: str = "en") -> str:
    """
    Run TranslationAgent on input text.
    If source_lang is None, detect language using LanguageDetectionAgent.
    Returns JSON string with translated text, detected source language, confidence, and reason.
    """

    # Step 1: Auto-detect source language if not provided
    if source_lang is None or source_lang.lower() == "auto":
        lang_result = LanguageDetectionAgent.run(text)
        source_lang = lang_result.language or "auto"

    # Step 2: Translate
    result = TranslationAgent.run(text, source_lang=source_lang, target_lang=target_lang)

    output = {
        "translated_text": result.text,
        "source_lang": result.source_lang,
        "target_lang": result.target_lang,
        "changed": result.changed,
        "confidence": result.confidence,
        "reason": result.reason
    }

    return output["translated_text"]
