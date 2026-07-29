import string
import math
from collections import Counter

ENGLISH_FREQ = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}

def clean_text(text: str) -> str:
    """Remove non-alpha characters and convert to uppercase."""
    return ''.join([c.upper() for c in text if c.isalpha()])

def chi_squared_score(text: str) -> float:
    """Calculate the chi-squared score of text against English letter frequencies."""
    text = clean_text(text)
    n = len(text)
    if n == 0:
        return 0.0
    
    counts = Counter(text)
    score = 0.0
    
    for char in string.ascii_uppercase:
        observed = counts[char]
        expected = n * ENGLISH_FREQ[char]
        if expected > 0:
            score += ((observed - expected) ** 2) / expected
            
    return score

def index_of_coincidence(text: str) -> float:
    """Calculate the Index of Coincidence (IC) for a given text."""
    text = clean_text(text)
    n = len(text)
    if n <= 1:
        return 0.0
        
    counts = Counter(text)
    ic_sum = sum(count * (count - 1) for count in counts.values())
    return ic_sum / (n * (n - 1))

def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy for the given text."""
    text = clean_text(text)
    n = len(text)
    if n == 0:
        return 0.0
        
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
        
    return entropy

def is_english(text: str, threshold: float = 0.5) -> bool:
    """Check if text looks like English using heuristic word matching or quadgrams.
    For basic usage, we check common English words and bigrams if spaces are omitted."""
    text = clean_text(text)
    if not text:
        return False
    
    # Common English words/bigrams that appear frequently
    common_ngrams = ["THE", "AND", "THA", "ENT", "ION", "TIO", "FOR", "NDE", "HAS", "NCE", "EDT", "TIS", "OFT", "STH"]
    matches = sum(1 for ngram in common_ngrams if ngram in text)
    
    # Also factor in IC since it's robust for English (around 0.0667)
    ic = index_of_coincidence(text)
    ic_valid = 0.055 < ic < 0.075
    
    score = (matches / len(common_ngrams)) + (0.5 if ic_valid else 0)
    return score > threshold

def gcd(a: int, b: int) -> int:
    """Greatest common divisor of a and b."""
    while b != 0:
        a, b = b, a % b
    return a

def mod_inverse(a: int, m: int) -> int:
    """Modular multiplicative inverse of a mod m."""
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return -1
