import math
from ..utils.text_utils import clean_text

def encrypt(plaintext: str, key: str) -> str:
    """Encrypts plaintext using a columnar transposition cipher."""
    plaintext = clean_text(plaintext)
    key = clean_text(key)
    if not key:
        return plaintext
    
    sorted_key = sorted(list(enumerate(key)), key=lambda x: x[1])
    col_order = [x[0] for x in sorted_key]
    
    col_len = len(key)
    row_len = math.ceil(len(plaintext) / col_len)
    
    padding = 'X' * ((row_len * col_len) - len(plaintext))
    padded_text = plaintext + padding
    
    grid = [padded_text[i:i+col_len] for i in range(0, len(padded_text), col_len)]
    
    ciphertext = []
    for col in col_order:
        for row in grid:
            ciphertext.append(row[col])
            
    return ''.join(ciphertext)

def decrypt(ciphertext: str, key: str) -> str:
    """Decrypts ciphertext using a columnar transposition cipher."""
    ciphertext = clean_text(ciphertext)
    key = clean_text(key)
    if not key:
        return ciphertext
        
    col_len = len(key)
    row_len = len(ciphertext) // col_len
    
    sorted_key = sorted(list(enumerate(key)), key=lambda x: x[1])
    
    cols = [''] * col_len
    for i, (orig_idx, _) in enumerate(sorted_key):
        start = i * row_len
        end = start + row_len
        cols[orig_idx] = ciphertext[start:end]
        
    plaintext = []
    for row in range(row_len):
        for col in range(col_len):
            plaintext.append(cols[col][row])
            
    return ''.join(plaintext)
