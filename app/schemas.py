from pydantic import BaseModel
from typing import Any, Dict

class EncryptRequest(BaseModel):
    message: str

class EncryptResponse(BaseModel):
    encrypted: str
    timestamp: str
    encryption_test: Dict[str, Any]

class DecryptRequest(BaseModel):
    encrypted: str

class DecryptResponse(BaseModel):
    message: str
    decryption_test: Dict[str, Any]
