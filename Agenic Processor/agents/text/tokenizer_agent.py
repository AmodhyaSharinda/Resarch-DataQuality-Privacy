# agents/text/tokenizer_agent.py

import nltk
from dataclasses import dataclass
from nltk.tokenize import word_tokenize

# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class TokenizerResult:
    tokens: list
    changed: bool
    confidence: float
    should_run: bool
    reason: str
    agent: str = "tokenizer_agent"
    error: str | None = None

class TokenizerAgent:

    @staticmethod
    def run(text: str) -> TokenizerResult:
        if not text or len(text.strip()) < 2:
            return TokenizerResult(
                tokens=[],
                changed=False,
                confidence=0.0,
                should_run=False,
                reason="Text too short to tokenize."
            )

        try:
            tokens = word_tokenize(text)
            changed = True if tokens else False
            confidence = 0.95 if tokens else 0.4
            reason = f"Tokenized into {len(tokens)} tokens." if tokens else "No tokens generated."

            return TokenizerResult(
                tokens=tokens,
                changed=changed,
                confidence=confidence,
                should_run=True,
                reason=reason
            )

        except Exception as e:
            return TokenizerResult(
                tokens=[],
                changed=False,
                confidence=0.0,
                should_run=False,
                reason="Tokenization failed.",
                error=str(e)
            )

# Legacy wrapper
def tokenize_text(text: str):
    result = TokenizerAgent.run(text)
    return result.tokens
