# agents/text/html_strip_agent.py

import re
from dataclasses import dataclass


@dataclass
class HtmlStripResult:
    text: str
    changed: bool
    confidence: float
    reason: str
    agent: str = "html_strip_agent"
    should_run: bool = False


class HTMLStripAgent:

    HTML_PATTERN = re.compile(r"<.*?>")

    @staticmethod
    def detect_html(text: str) -> bool:
        """Detect if the text contains HTML-like patterns."""
        return bool(HTMLStripAgent.HTML_PATTERN.search(text))

    @staticmethod
    def run(text: str) -> HtmlStripResult:
        """
        Agentic HTML cleaner:
        - Decides whether to run
        - Cleans HTML
        - Returns confidence + metadata
        """
        contains_html = HTMLStripAgent.detect_html(text)

        # If no HTML, skip the agent
        if not contains_html:
            return HtmlStripResult(
                text=text,
                changed=False,
                confidence=1.0,
                reason="No HTML patterns detected; skipping agent.",
                should_run=False
            )

        try:
            # Clean HTML
            cleaned = re.sub(r"<.*?>", "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            changed = cleaned != text

            return HtmlStripResult(
                text=cleaned,
                changed=changed,
                confidence=0.95 if changed else 0.8,
                reason="HTML tags removed successfully.",
                should_run=True
            )

        except Exception as e:
            # Agent fails safely
            return HtmlStripResult(
                text=text,
                changed=False,
                confidence=0.2,
                reason=f"HTML cleaning failed: {str(e)}",
                should_run=False
            )


# Convenient wrapper for backward compatibility
def clean_html(text: str):
    return HTMLStripAgent.run(text).text
