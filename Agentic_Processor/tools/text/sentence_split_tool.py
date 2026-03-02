# tools/sentence_split_tool.py

from langchain.tools import tool
from agents.text.sentence_split_agent import SentenceSplitAgent
import json

@tool("sentence_split_agent", description="Splits input text into sentences and returns them as JSON.")
def sentence_split_tool(text: str) -> str:
    """
    Run SentenceSplitAgent on input text.
    Returns a JSON string containing sentences, confidence, and metadata.
    """
    result = SentenceSplitAgent.run(text)

    output = {
        "sentences": result.sentences,
        "changed": result.changed,
        "confidence": result.confidence,
        "reason": result.reason
    }

    return "\n".join(result.sentences)
