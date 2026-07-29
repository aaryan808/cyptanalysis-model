from ..utils.text_utils import clean_text

def encrypt(plaintext: str, key: str) -> str:
    """Encrypts plaintext using the Vigenere cipher with a string keyword."""
    plaintext = clean_text(plaintext)
    key = clean_text(key)
    if not key:
        return plaintext
    
    ciphertext = []
    key_len = len(key)
    for i, char in enumerate(plaintext):
        p_val = ord(char) - 65
        k_val = ord(key[i % key_len]) - 65
        c_val = (p_val + k_val) % 26
        ciphertext.append(chr(c_val + 65))
    return ''.join(ciphertext)

def decrypt(ciphertext: str, key: str) -> str:
    """Decrypts ciphertext using the Vigenere cipher with a string keyword."""
    ciphertext = clean_text(ciphertext)
    key = clean_text(key)
    if not key:
        return ciphertext
    
    plaintext = []
    key_len = len(key)
    for i, char in enumerate(ciphertext):
        c_val = ord(char) - 65
        k_val = ord(key[i % key_len]) - 65
        p_val = (c_val - k_val) % 26
        plaintext.append(chr(p_val + 65))
    return ''.join(plaintext)
