# agents/text/validation_agent.py

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass
from typing import List, Dict, Any

class CleanedTextRecord(BaseModel):
    id: str
    source: str
    clean_text: str
    language: str
    sentences: List[str]
    tokens: List[str]
    confidence: float
    agents_run: List[str]
    errors: List[str]
    cleaning_status: str
    entities: List[Dict[str, Any]] = []

@dataclass
class ValidationResult:
    record: dict
    is_valid: bool
    confidence: float
    should_run: bool
    reason: str
    errors: List[str]
    agent: str = "validation_agent"

class ValidationAgent:

    cleaning_status = "cleaned"
    @staticmethod
    def run(record) -> ValidationResult:
        # Skip if record already marked failed
    
        if record.get("cleaning_status") == "failed":
            return ValidationResult(
                record=record,
                is_valid=False,
                confidence=0.0,
                should_run=False,
                reason="Record already marked as failed; skipping validation.",
                errors=record.get("errors", []),
            )

        try:
            validated = CleanedTextRecord(**record)
            return ValidationResult(
                record=record,
                is_valid=True,
                confidence=0.95,
                should_run=True,
                reason="Record validation succeeded.",
                errors=[]
            )
        except ValidationError as e:
            error_list = [err['msg'] for err in e.errors()]
            record['errors'] = record.get('errors', []) + error_list
            return ValidationResult(
                record=record,
                is_valid=False,
                confidence=0.0,
                should_run=True,
                reason="Record validation failed.",
                errors=error_list
            )

# Backwards compatible wrapper
def validate_record(record: dict):
    result = ValidationAgent.run(record)
    return result.record, result.is_valid
