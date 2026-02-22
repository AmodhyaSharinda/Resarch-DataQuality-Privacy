from google import genai
import json

class GeminiPlanner:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def plan(self, record):
        prompt = f"""
        You are an AI planner. Input record: {record}
        Decide which agents to run from:
        html_strip_agent, spell_agent, sentence_split_agent, tokenizer_agent,
        lang_detect_agent, translator_agent, ner_agent, validation_agent.
        Return ONLY a JSON array of agent names in the order they should run.
        """
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[{"text": prompt}]
        )
        plan_str = resp.text
        try:
            return json.loads(plan_str)
        except:
            # fallback plan if Gemini returns invalid JSON
            return ["spell_agent", "tokenizer_agent", "lang_detect_agent"]
