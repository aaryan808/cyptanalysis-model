import random
import math
import string

def _decrypt_with_matrix(ciphertext: str, matrix: list) -> str:
    def find_pos(char):
        idx = matrix.index(char)
        return idx // 5, idx % 5
        
    plaintext = ""
    for i in range(0, len(ciphertext), 2):
        if i + 1 >= len(ciphertext):
            plaintext += ciphertext[i]
            break
            
        a, b = ciphertext[i], ciphertext[i+1]
        if a == 'J': a = 'I'
        if b == 'J': b = 'I'
        
        if a not in matrix or b not in matrix:
            plaintext += a + b
            continue
            
        r1, c1 = find_pos(a)
        r2, c2 = find_pos(b)
        
        if r1 == r2:
            plaintext += matrix[r1 * 5 + (c1 - 1) % 5]
            plaintext += matrix[r2 * 5 + (c2 - 1) % 5]
        elif c1 == c2:
            plaintext += matrix[((r1 - 1) % 5) * 5 + c1]
            plaintext += matrix[((r2 - 1) % 5) * 5 + c2]
        else:
            plaintext += matrix[r1 * 5 + c2]
            plaintext += matrix[r2 * 5 + c1]
            
    return plaintext

def _generate_neighbor(matrix: list) -> list:
    new_matrix = matrix.copy()
    mutation = random.randint(0, 4)
    
    if mutation == 0:
        i, j = random.sample(range(25), 2)
        new_matrix[i], new_matrix[j] = new_matrix[j], new_matrix[i]
    elif mutation == 1:
        r1, r2 = random.sample(range(5), 2)
        for c in range(5):
            idx1, idx2 = r1 * 5 + c, r2 * 5 + c
            new_matrix[idx1], new_matrix[idx2] = new_matrix[idx2], new_matrix[idx1]
    elif mutation == 2:
        c1, c2 = random.sample(range(5), 2)
        for r in range(5):
            idx1, idx2 = r * 5 + c1, r * 5 + c2
            new_matrix[idx1], new_matrix[idx2] = new_matrix[idx2], new_matrix[idx1]
    elif mutation == 3:
        r = random.randint(0, 4)
        row = new_matrix[r*5:(r+1)*5]
        row.reverse()
        new_matrix[r*5:(r+1)*5] = row
    elif mutation == 4:
        c = random.randint(0, 4)
        col = [new_matrix[r*5+c] for r in range(5)]
        col.reverse()
        for r in range(5):
            new_matrix[r*5+c] = col[r]
            
    return new_matrix

def break_cipher(ciphertext: str, iterations: int = 10000, temperature: float = 20.0) -> dict:
    """Simulated annealing over 5x5 key matrices."""
    from src.utils.text_utils import clean_text
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    ciphertext = ciphertext.replace('J', 'I')
    scorer = QuadgramScorer()
    
    alphabet = list(string.ascii_uppercase.replace('J', ''))
    
    best_overall_score = float('-inf')
    best_overall_matrix = None
    best_overall_plaintext = ""
    
    for _ in range(5):
        current_matrix = alphabet.copy()
        random.shuffle(current_matrix)
        
        current_plaintext = _decrypt_with_matrix(ciphertext, current_matrix)
        current_score = scorer.score(current_plaintext)
        
        t = temperature
        cooling_rate = 0.9999
        
        for _ in range(iterations):
            new_matrix = _generate_neighbor(current_matrix)
            new_plaintext = _decrypt_with_matrix(ciphertext, new_matrix)
            new_score = scorer.score(new_plaintext)
            
            diff = new_score - current_score
            
            if diff > 0 or (t > 0 and random.random() < math.exp(diff / t)):
                current_score = new_score
                current_matrix = new_matrix
                current_plaintext = new_plaintext
                
            t *= cooling_rate
            
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_matrix = current_matrix
            best_overall_plaintext = current_plaintext
            
    return {
        "plaintext": best_overall_plaintext,
        "key": "".join(best_overall_matrix),
        "score": best_overall_score,
        "method": "simulated_annealing"
    }
