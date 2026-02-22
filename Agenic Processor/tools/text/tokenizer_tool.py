# tools/text/tokenizer_tool.py

from langchain.tools import tool
from agents.text.tokenizer_agent import TokenizerAgent
import json

@tool("tokenizer_agent", description="Tokenizes input text and returns tokens with metadata.")
def tokenizer_tool(text: str) -> str:
    """
    Runs TokenizerAgent on input text.
    Returns JSON string containing tokens, confidence, and reason.
    """
    result = TokenizerAgent.run(text)

    output = {
        "tokens": result.tokens,
        "changed": result.changed,
        "confidence": result.confidence,
        "reason": result.reason,
        "agent": result.agent
    }

    if result.error:
        output["error"] = result.error

    return json.dumps(output["tokens"], ensure_ascii=False)
