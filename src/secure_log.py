# secure_log.py
# -----------------------------------------------------------
# Secure encrypted logging for SQL Injection predictions
# - AES (Fernet) encryption with automatic key generation
# - Appends encrypted prediction logs
# - SHA256 hashing for tamper detection
# - Can decrypt and verify logs
# -----------------------------------------------------------

import os
import json
import base64
import datetime
import hashlib
from typing import Dict, List
from cryptography.fernet import Fernet, InvalidToken

# ---------------- Config (Project-specific paths) ----------------
BASE_DIR = r"C:\Users\HP\Downloads\DB_Sql_injection_proj"
KEY_PATH = os.path.join(BASE_DIR, "artifacts", "encryption.key")
LOG_PATH = os.path.join(BASE_DIR, "artifacts", "detections.log.enc")

# ---------------- Hashing Utility ----------------
def hash_record(record: Dict) -> str:
    """
    Compute a SHA256 hash of a record (used to verify tamper integrity).
    """
    record_json = json.dumps(record, sort_keys=True)
    return hashlib.sha256(record_json.encode("utf-8")).hexdigest()

# ---------------- Encryption Key Handling ----------------
def _load_or_create_key(path: str = KEY_PATH) -> bytes:
    """
    Loads the AES key or generates a new one if missing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    return key

def _cipher() -> Fernet:
    key = _load_or_create_key()
    return Fernet(key)

# ---------------- Core Logging Operations ----------------
def encrypt_and_append(record: Dict, log_path: str = LOG_PATH):
    """
    Encrypt and append a record (with optional hash) to the log file.
    """
    c = _cipher()

    # Add hash if not already present
    if "hash" not in record:
        record["hash"] = hash_record(record)

    payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
    token = c.encrypt(payload)

    with open(log_path, "ab") as f:
        f.write(token + b"\n")

def read_and_decrypt(log_path: str = LOG_PATH) -> List[Dict]:
    """
    Decrypt all encrypted log entries.
    Returns a list of dicts. Invalid lines are preserved as error entries.
    """
    c = _cipher()
    if not os.path.exists(log_path):
        return []

    out = []
    with open(log_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = c.decrypt(line)
                record = json.loads(payload)
                # Verify hash integrity if present
                if "hash" in record:
                    calc_hash = hash_record({k: v for k, v in record.items() if k != "hash"})
                    record["_verified"] = (calc_hash == record["hash"])
                else:
                    record["_verified"] = None
                out.append(record)
            except InvalidToken:
                out.append({"error": "INVALID_TOKEN", "raw": base64.b64encode(line).decode()})
    return out

# ---------------- Optional: Integrity Verification ----------------
def verify_log_integrity(log_path: str = LOG_PATH) -> Dict[str, int]:
    """
    Checks how many log records are verified, missing hashes, or tampered.
    Returns summary counts.
    """
    entries = read_and_decrypt(log_path)
    verified = sum(1 for e in entries if e.get("_verified") is True)
    tampered = sum(1 for e in entries if e.get("_verified") is False)
    nohash = sum(1 for e in entries if e.get("_verified") is None)
    return {"verified": verified, "tampered": tampered, "no_hash": nohash, "total": len(entries)}

# ---------------- Test / Standalone Run ----------------
if __name__ == "__main__":
    # Write a sample encrypted log entry
    test_record = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "query": "SELECT * FROM users WHERE id='1' OR '1'='1' --",
        "pred": 1,
        "proba": 0.9973
    }
    encrypt_and_append(test_record)

    # Read all logs and print verification summary
    entries = read_and_decrypt()
    print("Decrypted entries:")
    for e in entries:
        print(json.dumps(e, indent=2))

    print("\nIntegrity Summary:")
    print(verify_log_integrity())
