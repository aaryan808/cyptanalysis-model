import random
import string
from ..utils.text_utils import clean_text

def generate_random_key() -> str:
    """Generates a random 26-character permutation key for substitution cipher."""
    letters = list(string.ascii_uppercase)
    random.shuffle(letters)
    return ''.join(letters)

def encrypt(plaintext: str, key: str) -> str:
    """Encrypts plaintext using a monoalphabetic substitution cipher."""
    if len(set(key)) != 26 or len(key) != 26:
        raise ValueError("Key must be exactly 26 unique characters.")
    plaintext = clean_text(plaintext)
    key = key.upper()
    trans = str.maketrans(string.ascii_uppercase, key)
    return plaintext.translate(trans)

def decrypt(ciphertext: str, key: str) -> str:
    """Decrypts ciphertext using a monoalphabetic substitution cipher."""
    if len(set(key)) != 26 or len(key) != 26:
        raise ValueError("Key must be exactly 26 unique characters.")
    ciphertext = clean_text(ciphertext)
    key = key.upper()
    trans = str.maketrans(key, string.ascii_uppercase)
    return ciphertext.translate(trans)
