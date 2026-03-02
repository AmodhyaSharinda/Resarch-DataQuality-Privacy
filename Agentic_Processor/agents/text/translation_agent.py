# agents/text/translation_agent.py
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\Agenic Processor\keys\google_translate_key.json"

from dataclasses import dataclass
from google.cloud import translate_v2 as translate

translate_client = translate.Client()

@dataclass
class TranslationResult:
    text: str
    changed: bool
    confidence: float
    should_run: bool
    source_lang: str
    target_lang: str
    reason: str
    agent: str = "translator_agent"
    error: str | None = None

class TranslationAgent:

    @staticmethod
    def run(text: str, source_lang: str , target_lang: str = "en") -> TranslationResult:

        if not text or len(text.strip()) == 0:
            return TranslationResult(
                text=text,
                changed=False,
                confidence=0.1,
                should_run=False,
                source_lang=source_lang,
                target_lang=target_lang,
                reason="Text empty; skipping translation."
            )

        try:
            result = translate_client.translate(
                text,
                target_language=target_lang,
                source_language=None if source_lang == "auto" else source_lang,
                format_="text"
            )

            translated_text = result.get("translatedText", text)
            detected_lang = result.get("detectedSourceLanguage", source_lang)

            changed = translated_text != text
            confidence = 0.95 if changed else 1.0

            return TranslationResult(
                text=translated_text,
                changed=changed,
                confidence=confidence,
                should_run=True,
                source_lang=detected_lang,
                target_lang=target_lang,
                reason=f"Translated using Google Translate API from {detected_lang} to {target_lang}."
            )

        except Exception as e:
            return TranslationResult(
                text=text,
                changed=False,
                confidence=0.0,
                should_run=False,
                source_lang=source_lang,
                target_lang=target_lang,
                reason="Translation failed; returning original text.",
                error=str(e)
            )


def translate_to_english(text: str, source_lang: str) -> str:
    result = TranslationAgent.run(text, source_lang=source_lang, target_lang="en")
    return result.text




#pip install google-cloud-translate
if __name__ == "__main__":
    # Example non-English texts
    test_texts = [
        ("Dies ist ein deutscher Satz.", "de"),        # German
        ("这是一个中文句子。", "zh"),                    # Chinese
        ("Este es un texto en español.", "es"),        # Spanish
        ("This is already English.", "en"),           # English
        ("Ceci est une phrase française.", "fr")      # French
    ]

    for text, lang in test_texts:
        result = TranslationAgent.run(text, source_lang=lang, target_lang="en")
        print("--------------------------------------------------")
        print(f"Original text ({lang}): {text}")
        print(f"Detected source language: {result.source_lang}")
        print(f"Translated text: {result.text}")
        print(f"Changed: {result.changed}")
        print(f"Confidence: {result.confidence}")
        print(f"Reason: {result.reason}")
        if result.error:
            print(f"Error: {result.error}")