# tools/lang_detect_tool.py

from langchain.tools import tool
from agents.text.lang_detect_agent import LanguageDetectionAgent

@tool("lang_detect_agent", description="Detects the language of input text and returns a JSON with language code and confidence.")
def lang_detect_tool(text: str) -> str:
    """
    Run language detection using LanguageDetectionAgent.
    Returns JSON string: {"language": "en", "confidence": 0.99}
    """
    result = LanguageDetectionAgent.run(text)
    return f'{{"language": "{result.language}", "confidence": {result.confidence:.2f}}}'
