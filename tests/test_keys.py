# test_keys.py
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_public.pem"), "rb") as f:
    pub = serialization.load_pem_public_key(f.read())

print("Loaded key type:", type(pub))
print("Has verify method:", hasattr(pub, "verify"))