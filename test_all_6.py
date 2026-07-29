"""Final test script for hybrid classifier v7."""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter
import numpy as np
import math
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

def try_transposition_decrypt(ct, key_len):
    n = len(ct)
    if n < key_len or key_len < 2: return 0.0
    num_rows = math.ceil(n / key_len)
    col_len = n // key_len
    extra = n % key_len
    cols = []
    idx = 0
    for c in range(key_len):
        length = col_len + (1 if c < extra else 0)
        cols.append(ct[idx:idx+length])
        idx += length
    row_text = []
    for r in range(num_rows):
        for c in range(key_len):
            if r < len(cols[c]):
                row_text.append(cols[c][r])
    reconstructed = "".join(row_text)
    return count_english_word_chars(reconstructed) / n

def robust_classify(ciphertext, clf, ext):
    cleaned = re.sub(r'[^A-Z]', '', ciphertext.upper())
    n = len(cleaned)
    if n < 2: return "caesar", {}, "Fallback"

    feat_vec = ext.extract(ciphertext)
    probs = clf.predict_proba(feat_vec.reshape(1, -1))[0]
    classes = clf.pipeline.named_steps['classifier'].classes_
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    ml_pred = max(prob_dict, key=prob_dict.get)

    counts = Counter(cleaned)
    observed_freqs = sorted([counts.get(c, 0)/n for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], reverse=True)
    expected_freqs = sorted(ENGLISH_FREQ.values(), reverse=True)
    sorted_corr = np.corrcoef(observed_freqs, expected_freqs)[0, 1]

    if prob_dict.get('playfair', 0) > 0.35:
        return 'playfair', prob_dict, "ML (Playfair signature)"

    caesar_res = []
    for shift in range(1, 26):
        dec = try_caesar_decrypt(cleaned, shift)
        cov = count_english_word_chars(dec) / n
        chi = chi_squared_english(dec)
        caesar_res.append((shift, cov, chi))
    caesar_res.sort(key=lambda x: (-x[1], x[2]))
    best_caesar_shift, best_caesar_cov, best_caesar_chi = caesar_res[0]

    valid_a = [3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    affine_res = []
    for a in valid_a:
        a_inv = pow(a, -1, 26)
        for b in range(26):
            dec = try_affine_decrypt(cleaned, a, b)
            cov = count_english_word_chars(dec) / n
            chi = chi_squared_english(dec)
            affine_res.append(((a, b), cov, chi))
    affine_res.sort(key=lambda x: (-x[1], x[2]))
    best_affine_key, best_affine_cov, best_affine_chi = affine_res[0]

    best_trans_cov = 0.0
    for k in range(2, min(9, n)):
        cov = try_transposition_decrypt(cleaned, k)
        if cov > best_trans_cov:
            best_trans_cov = cov

    raw_cov = count_english_word_chars(cleaned) / n
    raw_chi = chi_squared_english(cleaned)

    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))

    best_pic = 0.0
    for period in range(2, min(16, n // 4 + 1)):
        cols = ['' for _ in range(period)]
        for i, c in enumerate(cleaned):
            cols[i % period] += c
        col_ics = [sum(v*(v-1) for v in Counter(col).values()) / (len(col)*(len(col)-1)) for col in cols if len(col) >= 4]
        if col_ics:
            avg = np.mean(col_ics)
            if avg > best_pic: best_pic = avg

    if best_caesar_cov >= 0.35 and best_caesar_cov >= best_affine_cov:
        return 'caesar', prob_dict, f"Probe: Caesar shift={best_caesar_shift} (word_cov={best_caesar_cov:.0%})"

    if best_affine_cov >= 0.35 and best_affine_cov > best_caesar_cov:
        return 'affine', prob_dict, f"Probe: Affine key={best_affine_key} (word_cov={best_affine_cov:.0%})"

    if raw_chi < 0.35 or raw_cov >= 0.25 or best_trans_cov >= 0.25 or prob_dict.get('columnar_transposition', 0) > 0.40:
        return 'columnar_transposition', prob_dict, f"Heuristic: raw_chi={raw_chi:.3f}, raw_cov={raw_cov:.0%}, trans_cov={best_trans_cov:.0%}"

    if (best_pic > 0.055 and best_pic > ic * 1.30) or (ic < 0.042 and n >= 70):
        return 'vigenere', prob_dict, f"Heuristic: periodic_IC={best_pic:.4f}, overall_IC={ic:.4f}"

    return 'substitution', prob_dict, f"Heuristic: sorted_freq_corr={sorted_corr:.2f}, IC={ic:.4f} (general substitution)"


print("Training ensemble classifier...")
clf = CipherClassifier(model_type='ensemble')
X, y = CipherClassifier.generate_training_data(n_samples=4000)
clf.train(X, y)
ext = FeatureExtractor()

print("\n" + "=" * 75)
print("  FINAL 6-WAY CLASSIFIER TEST RESULTS")
print("=" * 75)

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
    ("Long Substitution",
     substitution.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "QWERTYUIOPASDFGHJKLZXCVBNM"), "substitution"),
    ("Long Transposition (key=CASTLE)",
     columnar_transposition.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "CASTLE"), "columnar_transposition"),
    ("Long Playfair (key=KINGDOM)",
     playfair.encrypt("IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "KINGDOM"), "playfair"),
]

correct = 0
total = len(test_cases)
for name, ct, expected in test_cases:
    pred, pdict, method = robust_classify(ct, clf, ext)
    status = "PASS" if pred == expected else "FAIL"
    if pred == expected: correct += 1
    print(f"\n  [{status}] {name}")
    print(f"         CT: {ct[:40]}...")
    print(f"         Expected: {expected:<22} | Got: {pred}")
    print(f"         Method: {method}")

print(f"\n{'=' * 75}")
print(f"  OVERALL ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
print(f"{'=' * 75}")
