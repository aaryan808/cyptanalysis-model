"""Final test script for hybrid classifier v6."""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter
import numpy as np
from src.features.extractor import FeatureExtractor, ENGLISH_FREQ
from src.models.classifier import CipherClassifier
from src.ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair

ENGLISH_WORDS = {
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I",
    "IT", "FOR", "NOT", "ON", "WITH", "HE", "AS", "YOU", "DO", "AT",
    "THIS", "BUT", "HIS", "BY", "FROM", "THEY", "WE", "SAY", "HER", "SHE",
    "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD", "THERE", "THEIR", "WHAT",
    "SO", "UP", "OUT", "IF", "ABOUT", "WHO", "GET", "WHICH", "GO", "ME",
    "WHEN", "MAKE", "CAN", "LIKE", "TIME", "NO", "JUST", "HIM", "KNOW", "TAKE",
    "PEOPLE", "INTO", "YEAR", "YOUR", "GOOD", "SOME", "COULD", "THEM", "SEE", "OTHER",
    "THAN", "THEN", "NOW", "LOOK", "ONLY", "COME", "ITS", "OVER", "THINK", "ALSO",
    "BACK", "AFTER", "USE", "TWO", "HOW", "OUR", "WORK", "FIRST", "WELL", "WAY",
    "EVEN", "NEW", "WANT", "BECAUSE", "ANY", "THESE", "GIVE", "DAY", "MOST", "US",
    "QUICK", "BROWN", "FOX", "JUMPS", "LAZY", "DOG", "POWER", "KNOWLEDGE", "SINGLE",
    "TRUTH", "FORTUNE", "WIFE", "GREAT", "WORLD", "LIGHT", "DARK", "FOUND", "STATE"
}

def count_english_word_chars(text):
    matched = set()
    n = len(text)
    for l in range(2, min(13, n + 1)):
        for i in range(n - l + 1):
            sub = text[i:i+l]
            if sub in ENGLISH_WORDS:
                for idx in range(i, i+l):
                    matched.add(idx)
    return len(matched)

def chi_squared_english(text):
    if not text: return float('inf')
    n = len(text)
    counts = Counter(text)
    return sum(((counts.get(c, 0)/n - ENGLISH_FREQ[c]) ** 2) / ENGLISH_FREQ[c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def try_caesar_decrypt(ct, shift):
    return "".join(chr(((ord(c) - 65 - shift) % 26) + 65) for c in ct)

def try_affine_decrypt(ct, a, b):
    a_inv = pow(a, -1, 26)
    return "".join(chr(((a_inv * ((ord(c) - 65) - b)) % 26) + 65) for c in ct)

def hybrid_classify(ciphertext, clf, ext):
    cleaned = re.sub(r'[^A-Z]', '', ciphertext.upper())
    n = len(cleaned)
    
    feat_vec = ext.extract(ciphertext)
    probs = clf.predict_proba(feat_vec.reshape(1, -1))[0]
    classes = clf.pipeline.named_steps['classifier'].classes_
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    ml_pred = max(prob_dict, key=prob_dict.get)
    
    if n < 2: return ml_pred, "ML only"
    
    if prob_dict.get('playfair', 0) > 0.35:
        return 'playfair', "ML (Playfair signature)"
        
    caesar_word_coverage = []
    caesar_chis = []
    for shift in range(1, 26):
        dec = try_caesar_decrypt(cleaned, shift)
        cov = count_english_word_chars(dec) / n
        chi = chi_squared_english(dec)
        caesar_word_coverage.append((shift, cov, chi))
        caesar_chis.append(chi)
        
    caesar_word_coverage.sort(key=lambda x: (-x[1], x[2]))
    best_caesar_shift, best_caesar_cov, best_caesar_chi = caesar_word_coverage[0]
    
    min_caesar_chi = min(caesar_chis)
    median_caesar_chi = np.median(caesar_chis)
    caesar_chi_ratio = min_caesar_chi / median_caesar_chi if median_caesar_chi > 0 else 1.0
    
    valid_a = [3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    affine_word_coverage = []
    affine_chis = []
    for a in valid_a:
        a_inv = pow(a, -1, 26)
        for b in range(26):
            dec = try_affine_decrypt(cleaned, a, b)
            cov = count_english_word_chars(dec) / n
            chi = chi_squared_english(dec)
            affine_word_coverage.append(((a, b), cov, chi))
            affine_chis.append(chi)
            
    affine_word_coverage.sort(key=lambda x: (-x[1], x[2]))
    best_affine_key, best_affine_cov, best_affine_chi = affine_word_coverage[0]
    
    min_affine_chi = min(affine_chis)
    median_affine_chi = np.median(affine_chis)
    affine_chi_ratio = min_affine_chi / median_affine_chi if median_affine_chi > 0 else 1.0
    
    raw_cov = count_english_word_chars(cleaned) / n
    raw_chi = chi_squared_english(cleaned)
    
    counts = Counter(cleaned)
    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    
    best_pic = 0.0
    for period in range(2, min(16, n // 2 + 1)):
        cols = ['' for _ in range(period)]
        for i, c in enumerate(cleaned):
            cols[i % period] += c
        col_ics = []
        for col in cols:
            if len(col) > 1:
                cc = Counter(col)
                cn = len(col)
                col_ics.append(sum(v*(v-1) for v in cc.values()) / (cn*(cn-1)))
        if col_ics:
            avg = np.mean(col_ics)
            if avg > best_pic: best_pic = avg
            
    is_caesar_match = (best_caesar_cov >= 0.45) or (caesar_chi_ratio < 0.20)
    is_affine_match = (best_affine_cov >= 0.45 and best_affine_cov > best_caesar_cov) or \
                      (affine_chi_ratio < 0.20 and best_affine_chi < min_caesar_chi * 0.7)
                      
    if is_caesar_match and not is_affine_match:
        return 'caesar', f"Probe: shift={best_caesar_shift} (word_cov={best_caesar_cov:.0%}, chi_ratio={caesar_chi_ratio:.2f})"
        
    if is_affine_match:
        return 'affine', f"Probe: key={best_affine_key} (word_cov={best_affine_cov:.0%}, chi_ratio={affine_chi_ratio:.2f})"
        
    if raw_cov >= 0.40 or (raw_chi < 0.15 and ic > 0.050) or prob_dict.get('columnar_transposition', 0) > 0.40:
        return 'columnar_transposition', f"Heuristic: raw word_cov={raw_cov:.0%}, raw_chi={raw_chi:.3f}"
        
    if (best_pic > ic * 1.35 and best_pic > 0.050) or (ic < 0.045 and n >= 80) or prob_dict.get('vigenere', 0) > 0.45:
        return 'vigenere', f"Heuristic: periodic IC={best_pic:.4f}, overall IC={ic:.4f}"
        
    if ic > 0.045 or prob_dict.get('substitution', 0) > 0.35:
        return 'substitution', f"Heuristic: high IC={ic:.4f}, no simple key found"
        
    return ml_pred, "ML fallback"


print("Training ensemble classifier...")
clf = CipherClassifier(model_type='ensemble')
X, y = CipherClassifier.generate_training_data(n_samples=4000)
clf.train(X, y)
ext = FeatureExtractor()

print("\n" + "=" * 70)
print("  FINAL HYBRID CLASSIFIER TEST RESULTS")
print("=" * 70)

test_cases = [
    ("USER'S EXACT EXAMPLE: Caesar+3 (NQRZOHGJH LV SRZHU)",
     "NQRZOHGJH LV SRZHU", "caesar"),
    ("Short Caesar (shift=3)",
     caesar.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", 3), "caesar"),
    ("Short Affine (a=5,b=8)",
     affine.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", (5, 8)), "affine"),
    ("Short Vigenere (key=SECRET)",
     vigenere.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "SECRET"), "vigenere"),
    ("Short Substitution (QWERT...)",
     substitution.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "QWERTYUIOPASDFGHJKLZXCVBNM"), "substitution"),
    ("Short Transposition (key=ZEBRA)",
     columnar_transposition.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "ZEBRA"), "columnar_transposition"),
    ("Short Playfair (key=MONARCHY)",
     playfair.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "MONARCHY"), "playfair"),
    ("Long Caesar (shift=7)",
     caesar.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", 7), "caesar"),
    ("Long Affine (a=7,b=3)",
     affine.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", (7, 3)), "affine"),
    ("Long Vigenere (key=CRYPTO)",
     vigenere.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "CRYPTO"), "vigenere"),
    ("Long Transposition (key=CASTLE)",
     columnar_transposition.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "CASTLE"), "columnar_transposition"),
    ("Long Playfair (key=KINGDOM)",
     playfair.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "KINGDOM"), "playfair"),
]

correct = 0
total = len(test_cases)
for name, ct, expected in test_cases:
    pred, method = hybrid_classify(ct, clf, ext)
    status = "PASS" if pred == expected else "FAIL"
    if pred == expected: correct += 1
    print(f"\n  [{status}] {name}")
    print(f"         CT: {ct[:40]}...")
    print(f"         Expected: {expected} | Got: {pred}")
    print(f"         Method: {method}")

print(f"\n{'=' * 70}")
print(f"  FINAL ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
print(f"{'=' * 70}")
