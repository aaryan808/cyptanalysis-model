from ..utils.text_utils import clean_text, gcd, mod_inverse

def get_valid_a_values() -> list[int]:
    """Returns valid values for 'a' in the affine cipher (coprime to 26)."""
    return [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]

def encrypt(plaintext: str, key: tuple[int, int]) -> str:
    """Encrypts plaintext using the Affine cipher with a given key (a, b)."""
    a, b = key
    if gcd(a, 26) != 1:
        raise ValueError("Key 'a' must be coprime to 26")
    plaintext = clean_text(plaintext)
    ciphertext = []
    for char in plaintext:
        shifted = (a * (ord(char) - 65) + b) % 26
        ciphertext.append(chr(shifted + 65))
    return ''.join(ciphertext)

def decrypt(ciphertext: str, key: tuple[int, int]) -> str:
    """Decrypts ciphertext using the Affine cipher with a given key (a, b)."""
    a, b = key
    a_inv = mod_inverse(a, 26)
    if a_inv == -1:
        raise ValueError("Key 'a' has no modular inverse mod 26")
    ciphertext = clean_text(ciphertext)
    plaintext = []
    for char in ciphertext:
        shifted = (a_inv * ((ord(char) - 65) - b)) % 26
        plaintext.append(chr(shifted + 65))
    return ''.join(plaintext)
