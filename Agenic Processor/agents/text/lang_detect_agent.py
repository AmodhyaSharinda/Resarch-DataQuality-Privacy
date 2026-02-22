# agents/text/lang_detect_agent.py

from dataclasses import dataclass
import langid

@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    reason: str
    agent: str = "lang_detect_agent"
    should_run: bool = True
    error: str = None

class LanguageDetectionAgent:

    @staticmethod
    def is_text_valid(text: str) -> bool:
        """Check if text is long enough and contains letters."""
        return len(text.strip()) >= 3 and any(c.isalpha() for c in text)

    @staticmethod
    def run(text: str) -> LanguageDetectionResult:
        clean_text = text.strip()
        
        if not LanguageDetectionAgent.is_text_valid(clean_text):
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.1,
                reason="Text too short or contains no alphabetic characters.",
                should_run=False
            )

        try:
            lang, conf = langid.classify(clean_text)
            return LanguageDetectionResult(
                language=lang,
                confidence=conf,
                reason=f"Detected language '{lang}' with probability {conf:.2f}.",
                should_run=True
            )
        except Exception as e:
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.0,
                reason="Language detection failed.",
                error=str(e),
                should_run=False
            )

# Backwards compatibility wrapper
def detect_language(text: str):
    result = LanguageDetectionAgent.run(text)
    return result.language, result.confidence
