#NER1
import json
from transformers import XLMRobertaTokenizer, AutoModelForTokenClassification, pipeline

model_name = "Davlan/xlm-roberta-base-ner-hrl"

tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    ignore_labels=[]
)

def run_ner(text: str, min_confidence=0.6):
    raw = ner(text)
    results = []

    for e in raw:
        if e["score"] >= min_confidence and e["entity_group"] != "O":
            # Compute start/end manually
            value = e["word"]
            start = text.find(value)
            end = start + len(value)

            results.append({
                "entity": e["entity_group"],  # PER, LOC, ORG, etc.
                "value": value,
                "start": start,
                "end": end,
                "source": "ner",
                "confidence": round(float(e["score"]), 3)
            })

    return results


# Print all entity labels
labels = model.config.id2label
print(labels)


#NER2
import torch
from transformers import AutoModelForTokenClassification
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

MODEL_PATH = r"./xlmr_ner_model"

# ---- Load tokenizer.json DIRECTLY ----
raw_tokenizer = Tokenizer.from_file(
    f"{MODEL_PATH}/tokenizer.json"
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    sep_token="</s>",
    pad_token="<pad>",
    cls_token="<s>",
    mask_token="<mask>",
)

# ---- Load model ----
model2 = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)
model2.eval()

print("✅ Model & tokenizer loaded correctly")

def predict_ner(text):
    words = text.split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model2(**encoding)

    preds = outputs.logits.argmax(dim=-1)[0].cpu().numpy()

    results = []
    seen = set()

    for pred, wid in zip(preds, word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        results.append((words[wid], model2.config.id2label[pred]))

    return results

ENTITY_MAP = {
    "PERSON": "PER",
    "EMAIL": "EMAIL",
    "MARITAL_STATUS": "MARITAL_STATUS",
    "PASSWORD": "PASSWORD",
    "SEXUAL_ORIENTATION": "SEXUAL_ORIENTATION"
}

def bio_to_spans(tokens_and_labels, text, source="ner2"):
    entities = []
    current = None
    text_ptr = 0

    for token, label in tokens_and_labels:
        start = text.find(token, text_ptr)
        if start == -1:
            continue
        end = start + len(token)
        text_ptr = end

        if label.startswith("B-"):
            if current:
                entities.append(current)

            CONFIDENCE_FALLBACK = 0.85

            current = {
                "entity": label[2:],
                "value": token,
                "start": start,
                "end": end,
                "source": source,
                "confidence": CONFIDENCE_FALLBACK
            }


        elif label.startswith("I-") and current:
            current["value"] += " " + token
            current["end"] = end

        else:
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    return entities

# text = "Patient name Dinithi Rajapaksha has uf5676ADS and email dinithi@gmail.com. and she is a femal"

def run_ner2(text):
    tokens_labels = predict_ner(text)
    entities = bio_to_spans(tokens_labels, text)
    return entities

# print(run_ner2(text))
ENTITY_CANONICAL_MAP = {
    "PERSON": "PER",
    "PER": "PER",

    "LOCATION": "LOC",
    "LOC": "LOC",

    "ORGANIZATION": "ORG",
    "ORG": "ORG",
}

def normalize_entity_label(label: str) -> str:
    return ENTITY_CANONICAL_MAP.get(label.upper(), label.upper())



#NIC DETECTION
import re
from datetime import datetime
# =========================
# Regex patterns
# =========================
OLD_NIC_REGEX = re.compile(r"\b\d{9}[VXvx]\b")
NEW_NIC_REGEX = re.compile(r"\b\d{12}\b")
CURRENT_YEAR = datetime.now().year

# =========================
# Validation helpers
# =========================
def is_valid_day_of_year(day: int) -> bool:
    return (1 <= day <= 366) or (501 <= day <= 866)

def validate_old_nic(nic: str) -> bool:
    try:
        yy = int(nic[0:2])
        day = int(nic[2:5])

        # Day-of-year check (HARD)
        if not is_valid_day_of_year(day):
            return False

        # Infer full year (SOFT)
        inferred_year = 1900 + yy if yy > (CURRENT_YEAR % 100) else 2000 + yy

        # Year sanity check
        if inferred_year < 1900 or inferred_year > CURRENT_YEAR:
            return False

        return True

    except ValueError:
        return False



def validate_new_nic(nic: str) -> bool:
    try:
        # Handle 12 or 13 digit variants
        if len(nic) == 12:
            year = int(nic[0:4])
            day = int(nic[4:7])
        elif len(nic) == 13:
            year = int(nic[1:5])
            day = int(nic[5:8])
        else:
            return False

        # HARD checks
        if not is_valid_day_of_year(day):
            return False

        if not (1900 <= year <= CURRENT_YEAR):
            return False

        return True

    except ValueError:
        return False



# =========================
# Main NIC extractor
# =========================
def extract_nic(text: str):
    results = []

    # OLD NICs
    for match in OLD_NIC_REGEX.finditer(text):
        nic = match.group()
        if validate_old_nic(nic):
            results.append({
                "entity": "NIC",
                "value": nic,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    # NEW NICs
    for match in NEW_NIC_REGEX.finditer(text):
        nic = match.group()
        if validate_new_nic(nic):
            results.append({
                "entity": "NIC",
                "value": nic,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    return results

#CREDIR CARD DETECTION
import re

CREDIT_CARD_REGEX = re.compile(r"\b\d{13,19}\b")

def luhn_check(card_number: str) -> bool:
    total = 0
    reverse_digits = card_number[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)

        if i % 2 == 1:  # double every second digit
            n *= 2
            if n > 9:
                n -= 9

        total += n

    return total % 10 == 0

def valid_card_prefix(card_number: str) -> bool:
    length = len(card_number)

    # Visa
    if card_number.startswith("4") and length in (16, 19):
        return True

    # Mastercard
    if length in (16, 19):
        prefix2 = int(card_number[:2])
        prefix6 = int(card_number[:6])

        if 51 <= prefix2 <= 55:
            return True
        if 222100 <= prefix6 <= 272099:
            return True

    # American Express
    if length == 15 and card_number[:2] in ("34", "37"):
        return True

    # Discover (simplified but correct)
    if length in (14, 16):
        if card_number.startswith(("6011", "65")):
            return True
        if 622126 <= int(card_number[:6]) <= 623796:
            return True

    return False

def extract_credit_card(text: str):
    results = []

    for match in CREDIT_CARD_REGEX.finditer(text):
        token = match.group()

        try:
            if not valid_card_prefix(token):
                continue

            if not luhn_check(token):
                continue

            results.append({
                "entity": "CREDIT_CARD",
                "value": token,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })


        except ValueError:
            continue

    return results

#BANK ACCOUNT DETECTION
BANK_ACCOUNT_REGEX = re.compile(r"\b\d{10,18}\b")

VALID_BANK_ACCOUNT_LENGTHS = {
    11, 12, 13, 14, 15, 16, 18
}

KNOWN_BANK_PREFIXES = (
    "01", "10", "18", "88",   # Sampath / SCB
    "2042",                   # People's Bank
    "1", "2"                  # PABC
)


def extract_bank_account(text: str):
    results = []

    for match in BANK_ACCOUNT_REGEX.finditer(text):
        token = match.group()

        # Length validation (HARD)
        if len(token) not in VALID_BANK_ACCOUNT_LENGTHS:
            continue

        # Optional prefix check (SOFT)
        prefix_match = token.startswith(KNOWN_BANK_PREFIXES)

        results.append({
            "entity": "BANK_ACCOUNT",
            "value": token,
            "start": match.start(),
            "end": match.end(),
            "source": "regex",
            "confidence": 0.6 if prefix_match else 0.4
        })


    return results

#PHONE NUMBER DETECTION
import re

# --------------------------------------------------
# VALID CODES
# --------------------------------------------------

VALID_LANDLINE_AREA_CODES = {
    "011", "036", "031", "033", "038", "034",
    "054", "081", "051", "052", "066",
    "091", "041", "047",
    "032", "037",
    "021", "023", "024",
    "063", "067", "065", "026",
    "025", "027",
    "055", "057",
    "045", "035"
}

VALID_MOBILE_OPERATOR_CODES = {
    "070", "071", "072", "074",
    "075", "076", "077", "078"
}

# --------------------------------------------------
# REGEX (LOOSE MATCHING)
# --------------------------------------------------

PHONE_CANDIDATE_REGEX = re.compile(
    r"""
    (?:
        \+94|0
    )
    [\s\-]*
    \d{2}
    (?:[\s\-]*\d{3,4}){2}
    """,
    re.VERBOSE
)

# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_sri_lankan_number(raw: str) -> str:
    """
    Normalize Sri Lankan phone numbers to 0XXXXXXXXX
    """
    digits = re.sub(r"\D", "", raw)

    # +94XXXXXXXXX → 0XXXXXXXXX
    if digits.startswith("94") and len(digits) == 11:
        return "0" + digits[2:]

    # Already local format
    if digits.startswith("0") and len(digits) == 10:
        return digits

    return ""

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_landline_number(number: str) -> bool:
    return (
        len(number) == 10
        and number[:3] in VALID_LANDLINE_AREA_CODES
    )

def validate_mobile_number(number: str) -> bool:
    return (
        len(number) == 10
        and number[:3] in VALID_MOBILE_OPERATOR_CODES
    )

# --------------------------------------------------
# EXTRACTION (MAIN FUNCTION)
# --------------------------------------------------

def extract_phone_numbers(text: str):
    """
    Extracts Sri Lankan mobile & landline numbers from text
    """
    results = []

    for match in PHONE_CANDIDATE_REGEX.finditer(text):
        raw_value = match.group()
        normalized = normalize_sri_lankan_number(raw_value)

        if not normalized:
            continue

        if validate_mobile_number(normalized):
            phone_type = "MOBILE"
        elif validate_landline_number(normalized):
            phone_type = "LANDLINE"
        else:
            continue

        results.append({
            "entity": "PHONE_NUMBER",
            "value": raw_value,
            "start": match.start(),
            "end": match.end(),
            "source": "regex",
            "confidence": 1.0
        })

    return results

#IP ADDRESS DETECTION
import ipaddress

IPV4_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

IPV6_REGEX = re.compile(
    r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b"
)

def is_valid_ipv4(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False

    for part in parts:
        if not part.isdigit():
            return False
        if not 0 <= int(part) <= 255:
            return False

        # Prevent leading zeros like 001
        if part != "0" and part.startswith("0"):
            return False

    return True



def is_valid_ipv6(ip: str) -> bool:
    try:
        ipaddress.IPv6Address(ip)
        return True
    except ipaddress.AddressValueError:
        return False
    
    
    
def extract_ip_addresses(text: str):
    results = []

    # IPv4
    for match in IPV4_REGEX.finditer(text):
        token = match.group()
        if is_valid_ipv4(token):
            results.append({
                "entity": "IP_ADDRESS_V4",
                "value": token,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    # IPv6
    for match in IPV6_REGEX.finditer(text):
        token = match.group()
        if is_valid_ipv6(token):
            results.append({
                "entity": "IP_ADDRESS_V6",
                "value": token,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    return results

#MAC ADDRESS DETECTION
import re

MAC_COLON_HYPHEN_REGEX = re.compile(
    r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
)

MAC_DOT_REGEX = re.compile(
    r"\b(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}\b"
)

def is_valid_mac(mac: str) -> bool:
    # Normalize
    mac_clean = mac.replace(":", "").replace("-", "").replace(".", "")
    
    if len(mac_clean) != 12:
        return False
    
    if not all(c in "0123456789abcdefABCDEF" for c in mac_clean):
        return False
    
    return True


def get_mac_admin_type(mac: str) -> str:
    """
    Returns:
    - 'UAA' (Universally Administered Address)
    - 'LAA' (Locally Administered Address)
    """
    first_octet = mac.replace(":", "").replace("-", "").replace(".", "")[:2]
    first_octet_int = int(first_octet, 16)

    # Check U/L bit (bit 1)
    if first_octet_int & 0b00000010:
        return "LAA"
    else:
        return "UAA"

def extract_mac_addresses(text: str):
    results = []

    for match in MAC_COLON_HYPHEN_REGEX.finditer(text):
        token = match.group()
        if is_valid_mac(token):
            results.append({
                "entity": "MAC_ADDRESS",
                "value": token,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    for match in MAC_DOT_REGEX.finditer(text):
        token = match.group()
        if is_valid_mac(token):
            results.append({
                "entity": "MAC_ADDRESS",
                "value": token,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
                "confidence": 1.0
            })

    return results

#DATE DETECTION
import re
from datetime import datetime


DATE_CANDIDATE_REGEX = re.compile(
    r"""
    (
        # Numeric formats with separators (/ - . space)
        \b\d{1,4}[\s\-\/\.]\d{1,2}[\s\-\/\.]\d{1,4}\b
        |
        # Text month formats (5 January 2025)
        \b\d{1,2}\s+
        (Jan|January|Feb|February|Mar|March|Apr|April|May|
         Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|
         Nov|November|Dec|December)
        \s+\d{2,4}\b
        |
        # Text month formats (January 5, 2025)
        \b(Jan|January|Feb|February|Mar|March|Apr|April|May|
           Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|
           Nov|November|Dec|December)
        \s+\d{1,2},?\s+\d{2,4}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


DATE_FORMATS = [
    # Day first
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %m %Y",
    "%d/%m/%y", "%d-%m-%y", "%d %m %y",

    # Year first
    "%Y-%m-%d", "%Y/%m/%d", "%Y %m %d",

    # Textual
    "%d %B %Y", "%d %b %Y",
    "%B %d %Y", "%B %d, %Y",
    "%b %d %Y", "%b %d, %Y",
]

def normalize_date(raw: str) -> str:
    """
    Normalize dates to ISO format: YYYY-MM-DD
    """
    cleaned = re.sub(r"\s+", " ", raw.strip())

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return ""


def validate_date(normalized: str) -> bool:
    try:
        datetime.strptime(normalized, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def extract_dates(text: str):
    """
    Extracts dates from text in multiple formats
    """
    results = []

    for match in DATE_CANDIDATE_REGEX.finditer(text):
        raw_value = match.group()
        normalized = normalize_date(raw_value)

        if not normalized:
            continue

        if not validate_date(normalized):
            continue

        results.append({
            "entity": "DATE",
            "value": normalized,
            "raw": raw_value,
            "start": match.start(),
            "end": match.end(),
            "source": "regex",
            "confidence": 1.0
        })

    return results

def spans_overlap(a, b):
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])

ENTITY_PRIORITY = {
    "NIC": 100,
    "CREDIT_CARD": 100,
    "BANK_ACCOUNT": 90,
    "PHONE_NUMBER": 80,
    "IP_ADDRESS": 70,
    "MAC_ADDRESS": 70,

    # NER entities (lower priority)
    "PER": 30,
    "ORG": 30,
    "LOC": 30,
    "MISC": 20
}

def merge_entities(entities):
    """
    Resolves overlapping entities using priority + confidence.
    Deterministic rules win over NER.
    """

    merged = []

    for e in sorted(entities, key=lambda x: x["start"]):
        keep = True

        for m in merged:
            if spans_overlap(e, m):

                p_e = ENTITY_PRIORITY.get(e["entity"], 10)
                p_m = ENTITY_PRIORITY.get(m["entity"], 10)

                # Higher priority wins
                if p_e > p_m:
                    merged.remove(m)
                    continue

                # Same priority → higher confidence wins
                if p_e == p_m and e["confidence"] > m["confidence"]:
                    merged.remove(m)
                    continue

                # Otherwise discard current entity
                keep = False
                break

        if keep:
            merged.append(e)

    return merged

def normalize_entity_value(entity: str, value: str) -> str:
    value = value.strip()

    if entity in {"EMAIL", "PER", "ORG", "LOC"}:
        return value.lower()

    if entity in {"PHONE_NUMBER", "NIC", "CREDIT_CARD", "BANK_ACCOUNT"}:
        return re.sub(r"\D", "", value)

    return value

def deduplicate_by_highest_confidence(entities):
    """
    For the same (entity, value), keep ONLY the entity
    with the highest confidence.
    """
    best = {}

    for e in entities:
        key = (
            e["entity"],
            normalize_entity_value(e["entity"], e["value"])
        )

        if key not in best:
            best[key] = e
        else:
            if e["confidence"] > best[key]["confidence"]:
                best[key] = e

    return list(best.values())

def extract_pii(text: str):
    results = []

    # Regex-based entities
    results.extend(extract_nic(text))
    results.extend(extract_phone_numbers(text))
    results.extend(extract_ip_addresses(text))
    results.extend(extract_mac_addresses(text))
    results.extend(extract_dates(text))
    results.extend(extract_credit_card(text))
    results.extend(extract_bank_account(text))

    # NER entities
    results.extend(run_ner(text))
    results.extend(run_ner2(text))

    # 1️⃣ Normalize entity labels
    for r in results:
        r["entity"] = normalize_entity_label(r["entity"])

    # 2️⃣ Resolve span overlaps (priority-based)
    results = merge_entities(results)

    # 3️⃣ Deduplicate by highest confidence (FINAL)
    results = deduplicate_by_highest_confidence(results)

    return sorted(results, key=lambda x: x["start"])



# text = """
# {Amanda Arangalla, 200255701652, 0312220293, 0779381115, Negombo, Mega Trend Lanka (Pvt) Ltd, arangallaamanda@gmail.com}
# """

# entities = extract_pii(text)

# for e in entities:
#     print(e)

#LOAD POLICIES
def load_policy_map(policy_file_path):
    
    with open(policy_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    priority = {"High": 3, "Medium": 2, "Low": 1}
    policy_map = {}

    for policy in data["policies"]:
        entity = policy["entity"].strip().upper()
        level = policy["sensitivity_level"]

        # keep the highest sensitivity if duplicates exist
        if entity not in policy_map or priority[level] > priority[policy_map[entity]]:
            policy_map[entity] = level

    return policy_map

def add_sensitivity_levels(detected_entities, policy_map):
    """
    Adds sensitivity_level to each detected entity
    """
    enriched = []

    for ent in detected_entities:
        entity_type = ent["entity"].strip().upper()

        ent["sensitivity_level"] = policy_map.get(entity_type, "Unknown")
        enriched.append(ent)

    return enriched


# policy_map = load_policy_map("../policy_engine.json")
# entities = extract_pii(text)  # this returns the list of dicts as you showed

# # Step 3: Enrich entities with sensitivity_level
# enriched_entities = add_sensitivity_levels(entities, policy_map)
# enriched_entities

def find_sensitivity(text):
    entities = extract_pii(text)
    policy_map = load_policy_map("policy_engine.json")
    enriched_entities = add_sensitivity_levels(entities, policy_map)
    return enriched_entities

