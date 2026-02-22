from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

class OllamaPlanner:
    def __init__(self, model_name="text-llm"):
        self.model = OllamaLLM(model=model_name)

        # Define the template with input variable {record}
        self.template = """
            You are an AI planner. Input record: {record}
            Decide which agents to run from:
            html_strip_agent, spell_agent, sentence_split_agent, tokenizer_agent,
            lang_detect_agent, translator_agent, ner_agent, validation_agent.
            Return ONLY a JSON array of agent names in the order they should run.
            """
        # LangChain prompt
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Canonical order of agents
        self.correct_order = [
            "html_strip_agent",
            "lang_detect_agent",
            "translator_agent",
            "spell_agent",
            "sentence_split_agent",
            "tokenizer_agent",
            "ner_agent",
            "validation_agent"
        ]

    def plan(self, record):
    # Format the prompt
        formatted_prompt = self.prompt.format(record=record)
        
        # Call Ollama LLM
        response = self.model.invoke(formatted_prompt)

        # Extract raw text output
        resp = response.content if hasattr(response, "content") else str(response)

        # Parse JSON safely
        try:
            llm_plan = json.loads(resp)
            if not isinstance(llm_plan, list):
                raise ValueError
        except:
            llm_plan = ["spell_agent", "tokenizer_agent", "lang_detect_agent"]

        # Reorder according to canonical order
        ordered_plan = [agent for agent in self.correct_order if agent in llm_plan]

        return ordered_plan

# Example usage
if __name__ == "__main__":
    planner = OllamaPlanner()
    record = "This is a test input that needs processing."
    print(planner.plan(record))
