import string
import math
from collections import Counter

def kasiski_examination(ciphertext: str) -> list:
    """Find repeated trigrams and compute GCD of distances."""
    from src.utils.text_utils import clean_text
    ciphertext = clean_text(ciphertext)
    
    sequences = {}
    for i in range(len(ciphertext) - 2):
        seq = ciphertext[i:i+3]
        if seq not in sequences:
            sequences[seq] = []
        sequences[seq].append(i)
        
    distances = []
    for seq, positions in sequences.items():
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                distances.append(positions[i+1] - positions[i])
                
    if not distances:
        return []
        
    def get_factors(n):
        factors = []
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        factors.append(n)
        return factors
        
    factor_counts = Counter()
    for d in distances:
        for f in get_factors(d):
            factor_counts[f] += 1
            
    return [f for f, c in factor_counts.most_common() if 2 <= f <= 20]

def find_key_length(ciphertext: str, max_len: int = 20) -> int:
    """Use IC method to find key length."""
    from src.utils.text_utils import clean_text, index_of_coincidence
    ciphertext = clean_text(ciphertext)
    
    best_len = 1
    best_ic = 0
    
    for L in range(2, max_len + 1):
        avg_ic = 0
        for i in range(L):
            column = ciphertext[i::L]
            if len(column) > 1:
                avg_ic += index_of_coincidence(column)
        avg_ic /= L
        
        if avg_ic > best_ic:
            best_ic = avg_ic
            best_len = L
            
    return best_len

def find_key(ciphertext: str, key_length: int) -> str:
    from src.utils.text_utils import clean_text, chi_squared_score
    from src.ciphers.caesar import decrypt as caesar_decrypt
    
    ciphertext = clean_text(ciphertext)
    key = ""
    
    for i in range(key_length):
        column = ciphertext[i::key_length]
        
        best_shift = 0
        best_score = float('inf')
        
        for shift in range(26):
            decrypted_col = caesar_decrypt(column, shift)
            score = chi_squared_score(decrypted_col)
            if score < best_score:
                best_score = score
                best_shift = shift
                
        key += chr(best_shift + ord('A'))
        
    return key

def break_cipher(ciphertext: str, max_key_length: int = 20) -> dict:
    """Use Kasiski + IC to find key length, then frequency analysis."""
    from src.utils.text_utils import clean_text
    from src.ciphers.vigenere import decrypt
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    
    key_length = find_key_length(ciphertext, max_key_length)
    key = find_key(ciphertext, key_length)
    plaintext = decrypt(ciphertext, key)
    
    scorer = QuadgramScorer()
    score = scorer.score(plaintext)
    
    return {
        "plaintext": plaintext,
        "key": key,
        "score": score,
        "method": "ic_frequency_analysis"
    }
