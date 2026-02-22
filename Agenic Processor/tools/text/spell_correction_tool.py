# tools/spell_correction_tool.py

from langchain.tools import tool
from agents.text.spell_correction_agent import SpellCorrectionAgent
import json

@tool("spell_agent", description="Corrects spelling mistakes in English text and returns corrected text with metadata.")
def spell_tool(text: str) -> str:
    """
    Run SpellCorrectionAgent on input text.
    Returns JSON string containing corrected text, confidence, and number of words corrected.
    """
    result = SpellCorrectionAgent.run(text, language="en")

    output = {
        "corrected_text": result.text,
        "changed": result.changed,
        "confidence": result.confidence,
        "words_corrected": result.words_corrected,
        "reason": result.reason
    }

    return output["corrected_text"]
