"""Quick accuracy validation of the improved cipher classifier."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.models.classifier import CipherClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("=" * 60)
print("  CIPHER CLASSIFIER ACCURACY VALIDATION")
print("=" * 60)

# Generate 6000 training samples
print("\n[1/3] Generating 6000 training samples (6 cipher types)...")
X, y = CipherClassifier.generate_training_data(n_samples=6000)
print(f"      Dataset shape: X={X.shape}, y={y.shape}")
print(f"      Feature count: {X.shape[1]}")

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"      Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Train ensemble
print("\n[2/3] Training Voting Ensemble (RF + GradientBoosting + ExtraTrees)...")
clf = CipherClassifier(model_type='ensemble')
clf.train(X_train, y_train)

# Evaluate
print("\n[3/3] Evaluation on held-out test set:")
results = clf.evaluate(X_test, y_test)
print(results['classification_report'])

# Also test on specific known ciphertext
print("\n" + "=" * 60)
print("  LIVE CIPHER IDENTIFICATION TESTS")
print("=" * 60)

from src.features.extractor import FeatureExtractor
from src.ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair
import numpy as np

ext = FeatureExtractor()

test_cases = [
    ("Caesar (shift=3)", 
     caesar.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", 3), 
     "caesar"),
    ("Affine (a=5, b=8)", 
     affine.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", (5, 8)), 
     "affine"),
    ("Vigenere (key=SECRET)", 
     vigenere.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "SECRET"), 
     "vigenere"),
    ("Substitution (random key)", 
     substitution.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "QWERTYUIOPASDFGHJKLZXCVBNM"), 
     "substitution"),
    ("Columnar Transposition (key=ZEBRA)", 
     columnar_transposition.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "ZEBRA"), 
     "columnar_transposition"),
    ("Playfair (key=MONARCHY)", 
     playfair.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "MONARCHY"), 
     "playfair"),
]

correct = 0
for name, ciphertext, expected in test_cases:
    features = ext.extract(ciphertext).reshape(1, -1)
    pred = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]
    conf = max(probs)
    status = "PASS" if pred == expected else "FAIL"
    if pred == expected:
        correct += 1
    print(f"  {status} {name}: predicted={pred}, confidence={conf:.1%}")

print(f"\n  Live test accuracy: {correct}/{len(test_cases)}")
