# tools/validation_tool.py

from langchain.tools import tool
from agents.text.validation_agent import ValidationAgent
import json

@tool("validation_agent", description="Validates a cleaned text record for required fields and formats.")
def validation_tool(record: dict) -> str:
    """
    Runs ValidationAgent on a record dictionary.
    Returns JSON string: {"record": {...}, "is_valid": true/false, "reason": "..."}
    """
    result = ValidationAgent.run(record)
    
    output = {
        "record": result.record,
        "is_valid": result.is_valid,
        "confidence": result.confidence,
        "reason": result.reason,
        "errors": result.errors
    }
    
    return json.dumps(output, ensure_ascii=False)
