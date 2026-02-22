import uuid
import json
import os
import sys
from datetime import datetime

# LangChain imports
from kafka import KafkaProducer
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Add project root so Python can find agents/tools
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import planner
from agents.text.ollama_model import OllamaPlanner

# Import tools
from tools.text.html_strip_tool import html_strip_tool
from tools.text.spell_correction_tool import spell_tool
from tools.text.lang_detect_tool import lang_detect_tool
from tools.text.translation_tool import translation_tool
from tools.text.sentence_split_tool import sentence_split_tool
from tools.text.tokenizer_tool import tokenizer_tool
from tools.text.ner_tool import ner_tool
from tools.text.validation_tool import validation_tool

# All tools in a list
TOOLS = [
    html_strip_tool,
    lang_detect_tool,
    translation_tool,
    spell_tool,
    sentence_split_tool,
    tokenizer_tool,
    ner_tool,
    validation_tool,
]

class AgenticTextProcessor:
    def __init__(self):
        self.planner = OllamaPlanner()  # planner decides which agents to run

    def run(self, text: str, source="unknown"):
        record = {
            "id": str(uuid.uuid4()),
            "source": source,
            "clean_text": text,
            "language": "",
            "sentences": [],
            "tokens": [],
            "confidence": 0.0,
            "agents_run": [],
            "errors": [],
            "llm_plan": [],  # store LLM output
            "cleaning_status": "partial"
        }

        current_text = text
        try:
            # Planner decides which agents to run
            plan = self.planner.plan(current_text)
            record["llm_plan"].append(plan)
            print(f"[LLM PLAN] Text: {current_text}\nPlanned agents: {plan}\n")

            # Run only the agents decided by the planner
            for tool_name in plan:
                tool_obj = next((t for t in TOOLS if t.name == tool_name), None)
                if not tool_obj:
                    record["errors"].append(f"Tool not found: {tool_name}")
                    continue

                try:
                    output = tool_obj(text=current_text) if callable(tool_obj) else tool_obj.run(current_text)

                    # Print agent output
                    print(f"[AGENT OUTPUT] {tool_name} output:\n{output}\n")

                    # Handle special agents that save to separate fields
                    if tool_name == "tokenizer_agent":
                        record["tokens"] = output if isinstance(output, list) else json.loads(output)

                    elif tool_name == "lang_detect_agent":
                        # Parse language
                        record["language"] = output.get("language") if isinstance(output, dict) else json.loads(output).get("language")

                        # If language is not English, insert translator_agent as next in plan
                        if record["language"] != "en" and "translator_agent" not in plan:
                            current_index = plan.index(tool_name)
                            plan.insert(current_index + 1, "translator_agent")

                    elif tool_name == "ner_agent":
                        record["entity_detections"] = output if isinstance(output, list) else json.loads(output)
                    else:
                        # Update clean_text only for agents that transform text
                        if isinstance(output, str):
                            current_text = output
                            record["clean_text"] = current_text
                        elif isinstance(output, dict):
                            record["clean_text"] = output  # optional

                    record["agents_run"].append(tool_name)

                except Exception as e:
                    record["errors"].append(f"{tool_name} failed: {str(e)}")


            record["cleaning_status"] = "cleaned" if not record["errors"] else "failed"

            # attach timestamp for API/frontend
            record["timestamp"] = datetime.utcnow().isoformat()

            # send processed record to Kafka so API consumes it
            producer = KafkaProducer(
                bootstrap_servers="localhost:9092",
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )

            producer.send("processed_text", value=record)
            producer.flush()

            print("[TEXT ORCH] Sent to Kafka:", record["id"])

            # Save output
            output_file = "cleaned_output2.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_records, f, indent=2, ensure_ascii=False)

        except Exception as e:
            record["errors"].append(str(e))
            record["cleaning_status"] = "failed"

        
        return record



# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example record list
    records = [
        {"text": "Dies ist ein Beispieltext mit Fehlern, um zu entscheiden, welche Agenten ausgeführt werden sollen.",
        "source": "test_example"}
    ]

    processor = AgenticTextProcessor()
    cleaned_records = []

    for rec in records:
        for col, text in rec.items():
            if col != "source":
                cleaned = processor.run(text, source=rec.get("source", "unknown"))
                cleaned_records.append({
                    "column": col,
                    **cleaned
                })

    # Save output
    output_file = "cleaned_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Processed {len(cleaned_records)} text entries. Output saved to {output_file}")
