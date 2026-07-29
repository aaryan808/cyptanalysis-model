def break_cipher(ciphertext: str) -> dict:
    """Try all 312 valid (a,b) pairs, return best by quadgram score."""
    from src.utils.text_utils import clean_text
    from src.ciphers.affine import decrypt
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    scorer = QuadgramScorer()
    
    valid_a = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    
    best_score = float('-inf')
    best_plaintext = ""
    best_key = (1, 0)
    
    for a in valid_a:
        for b in range(26):
            try:
                plaintext = decrypt(ciphertext, (a, b))
                score = scorer.score(plaintext)
                if score > best_score:
                    best_score = score
                    best_plaintext = plaintext
                    best_key = (a, b)
            except Exception:
                pass
                
    return {
        "plaintext": best_plaintext,
        "key": best_key,
        "score": best_score,
        "method": "brute_force_quadgram"
    }
