import itertools
import random
import math

def _decrypt_with_permutation(ciphertext: str, key_length: int, permutation: list) -> str:
    num_rows = math.ceil(len(ciphertext) / key_length)
    num_empty = (num_rows * key_length) - len(ciphertext)
    
    cols = [''] * key_length
    col_lengths = [num_rows] * key_length
    
    empty_start = key_length - num_empty
    for i in range(empty_start, key_length):
        col_lengths[permutation.index(i)] -= 1
        
    idx = 0
    for i in range(key_length):
        orig_col = permutation[i]
        length = col_lengths[orig_col]
        cols[orig_col] = ciphertext[idx:idx+length]
        idx += length
        
    plaintext = ""
    for r in range(num_rows):
        for c in range(key_length):
            if r < len(cols[c]):
                plaintext += cols[c][r]
                
    return plaintext

def break_cipher(ciphertext: str, max_key_length: int = 10) -> dict:
    """Try key lengths 2-max_key_length, hill-climb column permutations."""
    from src.utils.text_utils import clean_text
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    scorer = QuadgramScorer()
    
    best_score = float('-inf')
    best_plaintext = ""
    best_key = None
    
    for length in range(2, max_key_length + 1):
        if length <= 6:
            for p in itertools.permutations(range(length)):
                p_list = list(p)
                plaintext = _decrypt_with_permutation(ciphertext, length, p_list)
                score = scorer.score(plaintext)
                if score > best_score:
                    best_score = score
                    best_plaintext = plaintext
                    best_key = p_list
        else:
            for _ in range(10):
                current_p = list(range(length))
                random.shuffle(current_p)
                
                current_plaintext = _decrypt_with_permutation(ciphertext, length, current_p)
                current_score = scorer.score(current_plaintext)
                
                for _ in range(1000):
                    i, j = random.sample(range(length), 2)
                    new_p = current_p.copy()
                    new_p[i], new_p[j] = new_p[j], new_p[i]
                    
                    new_plaintext = _decrypt_with_permutation(ciphertext, length, new_p)
                    new_score = scorer.score(new_plaintext)
                    
                    if new_score > current_score:
                        current_score = new_score
                        current_p = new_p
                        current_plaintext = new_plaintext
                        
                if current_score > best_score:
                    best_score = current_score
                    best_plaintext = current_plaintext
                    best_key = current_p
                    
    return {
        "plaintext": best_plaintext,
        "key": best_key,
        "score": best_score,
        "method": "brute_force_and_hill_climbing"
    }
