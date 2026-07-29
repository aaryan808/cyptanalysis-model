def identify_and_break(ciphertext: str, cipher_type: str = None) -> dict:
    """If cipher_type given, use that breaker. Otherwise return error asking to identify first."""
    if not cipher_type:
        return {"error": "Cipher type not provided. Please identify the cipher first."}
        
    try:
        cipher_type = cipher_type.lower()
        if cipher_type == 'caesar':
            from src.cryptanalysis.caesar_breaker import break_cipher
        elif cipher_type == 'affine':
            from src.cryptanalysis.affine_breaker import break_cipher
        elif cipher_type == 'vigenere':
            from src.cryptanalysis.vigenere_breaker import break_cipher
        elif cipher_type == 'substitution':
            from src.cryptanalysis.substitution_breaker import break_cipher
        elif cipher_type == 'transposition':
            from src.cryptanalysis.transposition_breaker import break_cipher
        elif cipher_type == 'playfair':
            from src.cryptanalysis.playfair_breaker import break_cipher
        else:
            return {"error": f"Unsupported cipher type: {cipher_type}"}
            
        return break_cipher(ciphertext)
    except Exception as e:
        return {"error": str(e)}
