# agents/mock_planner.py

class MockLLMPlanner:
    """
    Simulates GPT planning for agentic orchestrator.
    Returns a list of agent names in order to run for a given text record.
    Fully local, no API needed.
    """

    def plan(self, record):
        """
        record: dict containing at least "clean_text" key
        Returns: list of agent names in the order they should run
        """
        plan = []
        text = record.get("clean_text", "")

        # Heuristic rules to simulate GPT planning

        # 1. If HTML is present, run HTML stripping first
        if "<" in text or ">" in text:
            plan.append("html_strip_agent")

        # 2. Always run spell correction if text has letters
        if any(c.isalpha() for c in text):
            plan.append("spell_agent")

        # 3. Run sentence split if text has multiple sentences
        if "." in text or "!" in text or "?" in text:
            plan.append("sentence_split_agent")

        # 4. Run tokenizer (optional, generally after sentence split)
        plan.append("tokenizer_agent")

        # 5. Detect language
        plan.append("lang_detect_agent")

        # 6. Translate to English if not English
        plan.append("translator_agent")

        # 7. Named Entity Recognition
        if "@" in text or any(word.istitle() for word in text.split()):
            plan.append("ner_agent")

        # 8. Validate record at the end
        plan.append("validation_agent")

        # Remove duplicates while keeping order
        seen = set()
        final_plan = []
        for agent in plan:
            if agent not in seen:
                final_plan.append(agent)
                seen.add(agent)

        return final_plan
