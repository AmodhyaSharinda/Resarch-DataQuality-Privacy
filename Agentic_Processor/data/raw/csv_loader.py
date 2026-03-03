import csv
from typing import List, Dict

# Safe max field size for large CSV fields
csv.field_size_limit(10**9)

def load_csv(file_path: str, text_column: str = "text", source_column: str = "source") -> List[Dict]:
    """
    Load unstructured CSV into list of dicts, robust to encoding errors.
    """
    records = []
    try:
        with open(file_path, newline="", encoding="utf-8", errors="replace") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row.get(text_column, "").strip()
                source = row.get(source_column, "unknown").strip()
                if text:
                    records.append({"text": text, "source": source})
    except Exception as e:
        print(f"[CSV Loader Error]: {e}")

    return records
