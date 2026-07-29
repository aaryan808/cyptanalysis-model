import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from webapp.app import get_trained_classifier
from src.features.extractor import FeatureExtractor
from src.ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair
import numpy as np

print("Loading cached classifier (this will train it if not already cached)...")
clf = get_trained_classifier()
ext = FeatureExtractor()

test_cases = [
    # Short text (35 chars)
    ("Short Caesar", caesar.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", 3), 'caesar'),
    ("Short Affine", affine.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", (5, 8)), 'affine'),
    ("Short Vigenere", vigenere.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "SECRET"), 'vigenere'),
    ("Short Playfair", playfair.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "MONARCHY"), 'playfair'),
    
    # Long text (200 chars)
    ("Long Caesar", caesar.encrypt("ITISA TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE HOWEVER LITTLE KNOWN THE FEELINGS OR VIEWS OF SUCH A MAN MAY BE ON HIS FIRST ENTERING A", 3), 'caesar'),
    ("Long Affine", affine.encrypt("ITISA TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE HOWEVER LITTLE KNOWN THE FEELINGS OR VIEWS OF SUCH A MAN MAY BE ON HIS FIRST ENTERING A", (5, 8)), 'affine'),
    ("Long Vigenere", vigenere.encrypt("ITISA TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE HOWEVER LITTLE KNOWN THE FEELINGS OR VIEWS OF SUCH A MAN MAY BE ON HIS FIRST ENTERING A", "SECRET"), 'vigenere'),
    ("Long Playfair", playfair.encrypt("ITISA TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE HOWEVER LITTLE KNOWN THE FEELINGS OR VIEWS OF SUCH A MAN MAY BE ON HIS FIRST ENTERING A", "MONARCHY"), 'playfair'),
]

print("\n--- PREDICTION RESULTS ---")
for name, ct, expected in test_cases:
    vec = ext.extract(ct).reshape(1, -1)
    pred = clf.predict(vec)[0]
    probs = clf.predict_proba(vec)[0]
    conf = max(probs)
    print(f"{name}:")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {pred} (Confidence: {conf:.2f})")
    print("-" * 30)
