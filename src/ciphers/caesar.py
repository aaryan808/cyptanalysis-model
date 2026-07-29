from ..utils.text_utils import clean_text

def encrypt(plaintext: str, key: int) -> str:
    """Encrypts plaintext using the Caesar cipher with a given integer shift key."""
    plaintext = clean_text(plaintext)
    ciphertext = []
    for char in plaintext:
        shifted = (ord(char) - 65 + key) % 26
        ciphertext.append(chr(shifted + 65))
    return ''.join(ciphertext)

def decrypt(ciphertext: str, key: int) -> str:
    """Decrypts ciphertext using the Caesar cipher with a given integer shift key."""
    return encrypt(ciphertext, -key)
