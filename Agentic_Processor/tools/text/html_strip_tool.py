# tools/html_strip_tool.py

from langchain.tools import tool
from agents.text.html_strip_agent import HTMLStripAgent

html_agent = HTMLStripAgent()

@tool("html_strip_agent", description="Removes HTML tags from text")
def html_strip_tool(text: str) -> str:
    """
    Remove HTML tags from text and return cleaned output.
    """
    result = html_agent.run(text)
    return result.text
