import json

def extract_text_from_file(uploaded_file):
    """
    Extract text content from uploaded files.
    Supports: .txt, .json
    """
    if uploaded_file is None:
        return None

    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "txt":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "json":
            data = json.load(uploaded_file)

            # Convert JSON to readable text
            if isinstance(data, dict):
                return json.dumps(data, indent=2)
            elif isinstance(data, list):
                return json.dumps(data, indent=2)
            else:
                return str(data)

        else:
            return None

    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")
