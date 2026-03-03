# agents/text/sentence_split_agent.py

import nltk
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize

# Ensure Punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class SentenceSplitResult:
    sentences: list
    changed: bool
    confidence: float
    should_run: bool
    reason: str
    agent: str = "sentence_split_agent"
    error: str = None

class SentenceSplitAgent:

    @staticmethod
    def run(text: str) -> SentenceSplitResult:
        # Skip if text too short
        if not text or len(text.strip()) < 5:
            return SentenceSplitResult(
                sentences=[text],
                changed=False,
                confidence=0.0,
                should_run=False,
                reason="Text too short for meaningful sentence splitting."
            )

        try:
            text = text.strip('"')
            sentences = sent_tokenize(text)
            changed = True if sentences else False
            confidence = 0.95 if sentences else 0.4
            reason = "Sentence splitting executed successfully." if sentences else "No sentences found."

            return SentenceSplitResult(
                sentences=sentences,
                changed=changed,
                confidence=confidence,
                should_run=True,
                reason=reason
            )

        except Exception as e:
            return SentenceSplitResult(
                sentences=[],
                changed=False,
                confidence=0.0,
                should_run=False,
                reason="Sentence splitting failed.",
                error=str(e)
            )
