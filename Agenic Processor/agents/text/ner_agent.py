# agents/text/ner_agent.py

import spacy
from dataclasses import dataclass

# Load English NER model once
try:
    nlp = spacy.load("en_core_web_sm")
    MODEL_LOADED = True
except:
    MODEL_LOADED = False


@dataclass
class NERResult:
    entities: list
    confidence: float
    changed: bool
    reason: str
    agent: str = "ner_agent"
    should_run: bool = True
    error: str = None


class NERAgent:

    @staticmethod
    def is_text_valid(text: str) -> bool:
        """Requires minimal structure for NER to be meaningful."""
        return len(text.strip().split()) >= 3

    @staticmethod
    def is_model_available() -> bool:
        return MODEL_LOADED

    @staticmethod
    def run(text: str, language: str = "en") -> NERResult:
        """
        Agentic NER:
        - Checks language before running
        - Checks text validity
        - Returns confidence
        - Returns 'should_run'
        - Returns metadata for orchestrator reasoning
        """

        # ---------------------------
        # 1. Sanity checks
        # ---------------------------

        if not NERAgent.is_model_available():
            return NERResult(
                entities=[],
                confidence=0.0,
                changed=False,
                reason="SpaCy model not loaded.",
                should_run=False,
                error="ModelLoadError"
            )

        if language != "en":
            return NERResult(
                entities=[],
                confidence=0.1,
                changed=False,
                reason=f"NER skipped because text language is '{language}', not English.",
                should_run=False
            )

        if not NERAgent.is_text_valid(text):
            return NERResult(
                entities=[],
                confidence=0.2,
                changed=False,
                reason="Text too short or not structured enough for NER.",
                should_run=False
            )

        # ---------------------------
        # 2. Perform NER
        # ---------------------------
        try:
            doc = nlp(text)
            entities = [
                {
                    "type": ent.label_,
                    "text": ent.text
                    #"span": [ent.start_char, ent.end_char],
                    #"confidence": float(ent.kb_id_) if ent.kb_id_ else 0.85  # Fake fallback confidence
                }
                for ent in doc.ents
            ]

            confidence = 0.0
            if entities:
                confidence = sum(e["confidence"] for e in entities) / len(entities)
            else:
                confidence = 0.4  # No entities found but NER ran fine

            return NERResult(
                entities=entities,
                confidence=confidence,
                changed=True if entities else False,
                reason=f"Extracted {len(entities)} entities." if entities else "No entities detected.",
                should_run=True
            )

        except Exception as e:
            return NERResult(
                entities=[],
                confidence=0.1,
                changed=False,
                reason="NER processing failed.",
                should_run=False,
                error=str(e)
            )


# BACKWARD COMPATIBILITY WRAPPER
def extract_entities(text: str):
    result = NERAgent.run(text)
    return result.entities
