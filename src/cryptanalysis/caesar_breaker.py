import string

def break_cipher(ciphertext: str) -> dict:
    """Try all 26 shifts, return best decryption by quadgram score."""
    from src.utils.text_utils import clean_text
    from src.ciphers.caesar import decrypt
    from src.models.language_model import QuadgramScorer
    
    ciphertext = clean_text(ciphertext)
    scorer = QuadgramScorer()
    
    best_score = float('-inf')
    best_plaintext = ""
    best_key = 0
    
    for shift in range(26):
        plaintext = decrypt(ciphertext, shift)
        score = scorer.score(plaintext)
        if score > best_score:
            best_score = score
            best_plaintext = plaintext
            best_key = shift
            
    return {
        "plaintext": best_plaintext,
        "key": best_key,
        "score": best_score,
        "method": "brute_force_quadgram"
    }

def break_with_frequency(ciphertext: str) -> dict:
    from src.utils.text_utils import clean_text, chi_squared_score
    from src.ciphers.caesar import decrypt
    
    ciphertext = clean_text(ciphertext)
    
    best_score = float('inf')  # Lower chi-squared is better
    best_plaintext = ""
    best_key = 0
    
    for shift in range(26):
        plaintext = decrypt(ciphertext, shift)
        score = chi_squared_score(plaintext)
        if score < best_score:
            best_score = score
            best_plaintext = plaintext
            best_key = shift
            
    return {
        "plaintext": best_plaintext,
        "key": best_key,
        "score": best_score,
        "method": "frequency_chi_squared"
    }
