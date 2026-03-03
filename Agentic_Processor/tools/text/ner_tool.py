# tools/ner_tool.py

from langchain.tools import tool
from agents.text.ner_agent import NERAgent
import json

@tool("ner_agent", description="Extracts named entities from English text and returns them as JSON.")
def ner_tool(text: str) -> str:
    """
    Run NERAgent on input text.
    Returns a JSON string containing entities and confidence.
    """
    # Run the agent
    result = NERAgent.run(text, language="en")  # you could optionally pass detected language dynamically

    if not result.entities:  # empty list → False
        return json.dumps(text)

    
    # Convert result to JSON string
    output = {
        "entities": result.entities,
        "confidence": result.confidence,
        "changed": result.changed,
        "reason": result.reason
    }
    return json.dumps(output["entities"], ensure_ascii=False)
