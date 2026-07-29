from ..utils.text_utils import clean_text

def generate_key_matrix(keyword: str) -> list[list[str]]:
    """Generates a 5x5 key matrix for Playfair cipher using the given keyword."""
    keyword = clean_text(keyword).replace('J', 'I')
    seen = set()
    matrix_chars = []
    
    for char in keyword:
        if char not in seen:
            seen.add(char)
            matrix_chars.append(char)
            
    for char in "ABCDEFGHIKLMNOPQRSTUVWXYZ":
        if char not in seen:
            seen.add(char)
            matrix_chars.append(char)
            
    matrix = []
    for i in range(5):
        matrix.append(matrix_chars[i*5:(i+1)*5])
    return matrix

def _find_position(matrix: list[list[str]], char: str) -> tuple[int, int]:
    for r in range(5):
        for c in range(5):
            if matrix[r][c] == char:
                return r, c
    return -1, -1

def _prepare_text(text: str) -> str:
    text = clean_text(text).replace('J', 'I')
    prepared = []
    i = 0
    while i < len(text):
        c1 = text[i]
        if i + 1 < len(text):
            c2 = text[i+1]
            if c1 == c2:
                prepared.append(c1)
                prepared.append('X')
                i += 1
            else:
                prepared.append(c1)
                prepared.append(c2)
                i += 2
        else:
            prepared.append(c1)
            prepared.append('X')
            i += 1
            
    if len(prepared) % 2 != 0:
        prepared.append('X')
        
    return ''.join(prepared)

def encrypt(plaintext: str, key: str) -> str:
    """Encrypts plaintext using the Playfair cipher."""
    matrix = generate_key_matrix(key)
    prepared = _prepare_text(plaintext)
    ciphertext = []
    
    for i in range(0, len(prepared), 2):
        r1, c1 = _find_position(matrix, prepared[i])
        r2, c2 = _find_position(matrix, prepared[i+1])
        
        if r1 == r2:
            ciphertext.append(matrix[r1][(c1 + 1) % 5])
            ciphertext.append(matrix[r2][(c2 + 1) % 5])
        elif c1 == c2:
            ciphertext.append(matrix[(r1 + 1) % 5][c1])
            ciphertext.append(matrix[(r2 + 1) % 5][c2])
        else:
            ciphertext.append(matrix[r1][c2])
            ciphertext.append(matrix[r2][c1])
            
    return ''.join(ciphertext)

def decrypt(ciphertext: str, key: str) -> str:
    """Decrypts ciphertext using the Playfair cipher."""
    matrix = generate_key_matrix(key)
    ciphertext = clean_text(ciphertext).replace('J', 'I')
    plaintext = []
    
    if len(ciphertext) % 2 != 0:
        ciphertext += 'X'
        
    for i in range(0, len(ciphertext), 2):
        r1, c1 = _find_position(matrix, ciphertext[i])
        r2, c2 = _find_position(matrix, ciphertext[i+1])
        
        if r1 == r2:
            plaintext.append(matrix[r1][(c1 - 1) % 5])
            plaintext.append(matrix[r2][(c2 - 1) % 5])
        elif c1 == c2:
            plaintext.append(matrix[(r1 - 1) % 5][c1])
            plaintext.append(matrix[(r2 - 1) % 5][c2])
        else:
            plaintext.append(matrix[r1][c2])
            plaintext.append(matrix[r2][c1])
            
    return ''.join(plaintext)
