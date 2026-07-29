import random
import string
from collections import Counter

def _initial_key_from_frequency(ciphertext: str) -> dict:
    from src.utils.text_utils import clean_text, ENGLISH_FREQ
    ciphertext = clean_text(ciphertext)
    
    cipher_counts = Counter(ciphertext)
    cipher_sorted = [c for c, _ in cipher_counts.most_common()]
    
    for c in string.ascii_uppercase:
        if c not in cipher_sorted:
            cipher_sorted.append(c)
            
    english_sorted = sorted(ENGLISH_FREQ.keys(), key=lambda k: ENGLISH_FREQ[k], reverse=True)
    
    key = {}
    for c, e in zip(cipher_sorted, english_sorted):
        key[c] = e
        
    return key

def break_cipher(ciphertext: str, max_iterations: int = 5000, num_restarts: int = 20) -> dict:
    """Hill-climbing with random restarts using quadgram scoring."""
    from src.utils.text_utils import clean_text
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    scorer = QuadgramScorer()
    
    best_overall_score = float('-inf')
    best_overall_key = None
    best_overall_plaintext = ""
    
    for _ in range(num_restarts):
        current_key = _initial_key_from_frequency(ciphertext)
        
        def decrypt(ct, k):
            return "".join(k.get(c, c) for c in ct)
            
        current_plaintext = decrypt(ciphertext, current_key)
        current_score = scorer.score(current_plaintext)
        
        for _ in range(max_iterations):
            a, b = random.sample(list(string.ascii_uppercase), 2)
            
            new_key = current_key.copy()
            new_key[a], new_key[b] = new_key[b], new_key[a]
            
            new_plaintext = decrypt(ciphertext, new_key)
            new_score = scorer.score(new_plaintext)
            
            if new_score > current_score:
                current_score = new_score
                current_key = new_key
                current_plaintext = new_plaintext
                
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_key = current_key
            best_overall_plaintext = current_plaintext
            
    return {
        "plaintext": best_overall_plaintext,
        "key": best_overall_key,
        "score": best_overall_score,
        "method": "hill_climbing"
    }
