# agents/text/spell_correction_agent.py

from dataclasses import dataclass
from spellchecker import SpellChecker

spell = SpellChecker()

@dataclass
class SpellCorrectionResult:
    text: str
    changed: bool
    confidence: float
    should_run: bool
    words_corrected: int
    reason: str
    agent: str = "spell_agent"
    error: str | None = None

class SpellCorrectionAgent:

    @staticmethod
    def estimate_noise(text: str) -> float:
        tokens = text.split()
        if not tokens:
            return 0.0
        misspelled = spell.unknown(tokens)
        return len(misspelled) / len(tokens)

    @staticmethod
    def run(text: str, language: str = "en") -> SpellCorrectionResult:

        # -----------------------------
        # 1. Should spell correction run?
        # -----------------------------
        if language != "en":
            return SpellCorrectionResult(
                text=text,
                changed=False,
                confidence=0.1,
                should_run=False,
                words_corrected=0,
                reason=f"Spell correction skipped because language is '{language}'."
            )

        if len(text.strip().split()) < 2:
            return SpellCorrectionResult(
                text=text,
                changed=False,
                confidence=0.1,
                should_run=False,
                words_corrected=0,
                reason="Text too short for spell correction."
            )

        noise = SpellCorrectionAgent.estimate_noise(text)
        if noise < 0.10:  # Less than 10% noisy
            return SpellCorrectionResult(
                text=text,
                changed=False,
                confidence=0.90,
                should_run=False,
                words_corrected=0,
                reason="Text appears clean; spell correction not needed."
            )

        # -----------------------------
        # 2. Perform spell correction
        # -----------------------------
        try:
            words = text.split()
            corrected_words = []
            corrected_count = 0

            for w in words:
                if not w.isalpha():  # skip digits/special chars
                    corrected_words.append(w)
                    continue
                corrected = spell.correction(w)
                if corrected != w:
                    corrected_count += 1
                corrected_words.append(corrected)

            corrected_text = " ".join(corrected_words)

            return SpellCorrectionResult(
                text=corrected_text,
                changed=(corrected_count > 0),
                confidence=0.85 if corrected_count > 0 else 0.40,
                should_run=True,
                words_corrected=corrected_count,
                reason=f"Corrected {corrected_count} words."
            )

        except Exception as e:
            return SpellCorrectionResult(
                text=text,
                changed=False,
                confidence=0.0,
                should_run=False,
                words_corrected=0,
                reason="Spell correction failed.",
                error=str(e)
            )

# BACKWARD COMPATIBILITY
def correct_spelling(text: str):
    result = SpellCorrectionAgent.run(text)
    return result.text
