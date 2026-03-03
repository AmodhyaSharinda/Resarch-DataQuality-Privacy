import os
import uuid
import base64
import hashlib
import bcrypt
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# CONFIGURATION
# ============================================================

SENSITIVITY_MAP = {
    "LOW": 1,
    "MEDIUM": 3,
    "HIGH": 5
}

CONTEXT_MAP = {
    "logging": 1,
    "storage": 2,
    "analytics": 3,
    "external_transfer": 5
}

# AES 256-bit key (In production, store in KMS!)
AES_KEY = AESGCM.generate_key(bit_length=256)

DEFAULT_SALT = "my_secure_salt"

# Simulated token vault (Use DB in production)
token_vault = {}


# ============================================================
# RISK SCORE
# RS = 0.7S + 0.3C
# ============================================================

def calculate_risk_score(sensitivity_level: str, data_usage_context: str) -> float:
    S = SENSITIVITY_MAP.get(sensitivity_level.upper(), 1)
    C = CONTEXT_MAP.get(data_usage_context.lower(), 1)
    return 0.7 * S + 0.3 * C


# ============================================================
# AES-256-GCM ENCRYPTION
# ============================================================

def aes_encrypt(plaintext: str) -> str:
    aesgcm = AESGCM(AES_KEY)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ciphertext).decode()


def aes_decrypt(ciphertext_b64: str) -> str:
    data = base64.b64decode(ciphertext_b64)
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(AES_KEY)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value: str) -> str:
    token = "TKN_" + uuid.uuid4().hex[:12]
    token_vault[token] = value
    return token


def detokenize(token: str) -> str:
    return token_vault.get(token)


# ============================================================
# HASHING (SHA-256)
# ============================================================

def hash_value(value: str) -> str:
    combined = (DEFAULT_SALT + value).encode()
    return hashlib.sha256(combined).hexdigest()


# ============================================================
# PASSWORD HASHING (BCRYPT)
# ============================================================

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


# ============================================================
# MASKING
# ============================================================

def partial_mask(value: str, visible_start: int = 2, visible_end: int = 2) -> str:
    if len(value) <= visible_start + visible_end:
        return "*" * len(value)

    return (
        value[:visible_start]
        + "*" * (len(value) - visible_start - visible_end)
        + value[-visible_end:]
    )


# ============================================================
# GENERALIZATION
# ============================================================

def generalize_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return str(dt.year)


def generalize_location(location: str) -> str:
    parts = location.split(",")
    return parts[-1].strip() if len(parts) > 1 else location


# ============================================================
# MAIN PROTECTION ENGINE
# ============================================================

# def apply_protection(
#     value: str,
#     sensitivity_level: str,
#     data_usage_context: str,
#     entity_type: str = None
# ):
#     """
#     Main protection function
#     Returns only required fields to frontend
#     """

#     # Handle empty value safely
#     if not value:
#         return {
#             "entity_type": entity_type,
#             "sensitivity_level": sensitivity_level,
#             "data_usage_context": data_usage_context,
#             "risk_score": 0,
#             "method_used": "NONE",
#             "protected_value": value
#         }

#     # Special rule: PASSWORD always bcrypt
#     if entity_type and entity_type.upper() == "PASSWORD":
#         protected_value = hash_password(value)
#         return {
#             "entity_type": entity_type,
#             "sensitivity_level": sensitivity_level,
#             "data_usage_context": data_usage_context,
#             "risk_score": 5.0,  # treat as maximum risk
#             "method_used": "BCRYPT_PASSWORD_HASH",
#             "protected_value": protected_value
#         }

#     # Calculate risk score
#     rs = calculate_risk_score(sensitivity_level, data_usage_context)

#     # Decision thresholds
#     if rs <= 2:
#         method = "MASKING"
#         protected_value = partial_mask(value)

#     elif 2 < rs <= 3.5:
#         method = "HASH_SHA256"
#         protected_value = hash_value(value)

#     elif 3.5 < rs <= 4.5:
#         method = "AES_256_GCM"
#         protected_value = aes_encrypt(value)

#     else:
#         method = "TOKENIZATION"
#         protected_value = tokenize(value)

#     return {
#         "entity_type": entity_type,
#         "sensitivity_level": sensitivity_level,
#         "data_usage_context": data_usage_context,
#         "risk_score": round(rs, 2),
#         "method_used": method,
#         "protected_value": protected_value
#     }

# ============================================================
# METHOD SELECTION ENGINE (NO TRANSFORMATION)
# ============================================================

def select_protection_method(
    entity_type: str = None,
    risk_score: float = None
):
    """
    Decides which protection method SHOULD be used
    based on risk_score.
    Does NOT calculate risk score.
    """

    # Special rule: PASSWORD always bcrypt
    if entity_type and entity_type.upper() == "PASSWORD":
        return {
            "entity_type": entity_type,
            "suggested_method": "BCRYPT_PASSWORD_HASH"
        }

    if risk_score is None:
        raise ValueError("risk_score must be provided")

    # Select method based on risk score
    if risk_score <= 2:
        method = "MASKING" 

    elif 2 < risk_score <= 3.5:
        method = "HASH_SHA256"

    elif 3.5 < risk_score <= 4.5:
        method = "AES_256_GCM"

    else:
        method = "TOKENIZATION"

    return {
        "entity_type": entity_type,
        "suggested_method": method
    }



# ============================================================
# APPLY SELECTED PROTECTION
# ============================================================

def apply_selected_protection(value: str, method: str):
    """
    Applies the method selected by user.
    """

    if not value:
        return value

    if method == "MASKING":
        return partial_mask(value)

    elif method == "HASH_SHA256":
        return hash_value(value)

    elif method == "AES_256_GCM":
        return aes_encrypt(value)

    elif method == "TOKENIZATION":
        return tokenize(value)

    elif method == "BCRYPT_PASSWORD_HASH":
        return hash_password(value)

    else:
        return value