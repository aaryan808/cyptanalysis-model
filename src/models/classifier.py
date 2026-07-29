"""
High-accuracy ML Classifier for Classical Cipher Identification.

Key improvements for accuracy:
  - Uses actual cipher module encrypt() for training data (not simplified inline versions)
  - Generates 6000+ samples from a large diverse English corpus
  - Uses GradientBoosting + ExtraTrees voting ensemble for robust predictions
  - 60+ engineered features specifically designed to discriminate cipher families
"""

import numpy as np
import random
import string
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from ..features.extractor import FeatureExtractor


class CipherClassifier:
    CIPHER_TYPES = ['caesar', 'affine', 'vigenere', 'substitution',
                    'columnar_transposition', 'playfair']
    
    def __init__(self, model_type='ensemble'):
        """Initialize classifier.
        
        Args:
            model_type: 'ensemble' (recommended), 'random_forest', 'gradient_boosting',
                        'extra_trees', 'svm', or 'neural_network'
        """
        self.model_type = model_type
        
        if model_type == 'ensemble':
            # Voting ensemble of 3 strong classifiers
            rf = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_split=3,
                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
            )
            gb = GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            et = ExtraTreesClassifier(
                n_estimators=400, max_depth=None, min_samples_split=2,
                random_state=42, n_jobs=-1
            )
            clf = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('et', et)],
                voting='soft'
            )
        elif model_type == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_split=2,
                random_state=42, n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            clf = GradientBoostingClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        elif model_type == 'extra_trees':
            clf = ExtraTreesClassifier(
                n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
            )
        elif model_type == 'svm':
            clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        elif model_type == 'neural_network':
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), max_iter=1000,
                early_stopping=True, random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        self.extractor = FeatureExtractor()

    def train(self, X, y):
        """Train the classifier on feature matrix X and labels y."""
        self.pipeline.fit(X, y)

    def predict(self, X):
        """Predict cipher type."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Return probability distribution over cipher types."""
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Return classification report and confusion matrix."""
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return {'classification_report': report, 'confusion_matrix': conf_matrix}

    def cross_validate(self, X, y, cv=5):
        """Run stratified k-fold cross-validation and return scores."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(
            self.pipeline, X, y, cv=skf,
            scoring='accuracy', return_train_score=True
        )
        return {
            'test_accuracy_mean': scores['test_score'].mean(),
            'test_accuracy_std': scores['test_score'].std(),
            'train_accuracy_mean': scores['train_score'].mean(),
        }

    def save(self, filepath):
        """Save trained model to file."""
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath):
        """Load trained model from file."""
        self.pipeline = joblib.load(filepath)

    def get_feature_importance(self):
        """Return feature importance (for tree-based models)."""
        clf = self.pipeline.named_steps['classifier']
        # Handle VotingClassifier by using the first estimator's importances
        if hasattr(clf, 'estimators_'):
            for est in clf.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances = est.feature_importances_
                    break
            else:
                raise ValueError("No estimator with feature_importances_ found.")
        elif hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            raise ValueError("Feature importance not available for this model type.")
        feature_names = self.extractor.get_feature_names()
        return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------ #
    #  LARGE ENGLISH CORPUS for training data generation                  #
    # ------------------------------------------------------------------ #
    _CORPUS = """
It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife however little known the feelings or views of such a man may be on his first entering a neighbourhood this truth is so well fixed in the minds of the surrounding families that he is considered the rightful property of some one or other of their daughters my dear mister bennet said his lady to him one day have you heard that netherfield park is let at last mister bennet replied that he had not but it is returned she for missus long has just been here and she told me all about it mister bennet made no answer do you not want to know who has taken it cried his wife impatiently you want to tell me and i have no objection to hearing it this was invitation enough why my dear you must know missus long says that netherfield is taken by a young man of large fortune from the north of england that he came down on monday in a chaise and four to see the place and was so much delighted with it that he agreed with mister morris immediately that he is to take possession before michaelmas and some of his servants are to be in the house by the end of next week what is his name bingley is he married or single oh single my dear to be sure a single man of large fortune four or five thousand a year what a fine thing for our girls how so how can it affect them my dear mister bennet replied his wife how can you be so tiresome you must know that i am thinking of his marrying one of them is that his design in settling here design nonsense how can you talk so but it is very likely that he may fall in love with one of them and therefore you must visit him as soon as he comes i see no occasion for that you and the girls may go or you may send them by themselves which perhaps will be still better for as you are as handsome as any of them mister bingley may like you the best of the party

Call me ishmael some years ago never mind how long precisely having little or no money in my purse and nothing particular to interest me on shore i thought i would sail about a little and see the watery part of the world it is a way i have of driving off the spleen and regulating the circulation whenever i find myself growing grim about the mouth whenever it is a damp drizzly november in my soul whenever i find myself involuntarily pausing before coffin warehouses and bringing up the rear of every funeral i meet and especially whenever my hypos get such an upper hand of me that it requires a strong moral principle to prevent me from deliberately stepping into the street and methodically knocking peoples hats off then i account it high time to get to sea as soon as i can this is my substitute for pistol and ball with a philosophical flourish cato throws himself upon his sword i quietly take to the ship

It was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness it was the epoch of belief it was the epoch of incredulity it was the season of light it was the season of darkness it was the spring of hope it was the winter of despair we had everything before us we had nothing before us we were all going direct to heaven we were all going direct the other way in short the period was so far like the present period that some of its noisiest authorities insisted on its being received for good or for evil in the superlative degree of comparison only

In the beginning god created the heaven and the earth and the earth was without form and void and darkness was upon the face of the deep and the spirit of god moved upon the face of the waters and god said let there be light and there was light and god saw the light that it was good and god divided the light from the darkness and god called the light day and the darkness he called night and the evening and the morning were the first day

The study of classical ciphers and their cryptanalysis provides an essential foundation for understanding modern encryption techniques. Monoalphabetic substitution ciphers preserve the frequency distribution of the plaintext while simply remapping letters according to a fixed permutation. The Caesar cipher is the simplest example where each letter is shifted by a constant amount. The Affine cipher generalizes this with a linear function. In contrast polyalphabetic ciphers like the Vigenere cipher use multiple alphabets destroying the simple frequency signature of the plaintext. The Index of Coincidence drops from about zero point zero six seven for English to near zero point zero three eight for a well encrypted Vigenere text. Transposition ciphers rearrange the positions of plaintext letters without changing their identities so the overall frequency distribution remains identical to standard English. The Playfair cipher operates on pairs of letters using a five by five matrix and produces ciphertext that never contains double letters in its digraphs.

Machine learning approaches to cryptanalysis leverage statistical features extracted from ciphertext to automatically identify the encryption method used. Key discriminative features include the index of coincidence which measures how likely it is that two randomly chosen letters from the text are the same. For standard english text this value is approximately zero point zero six seven while for randomly generated text it drops to one over twenty six or about zero point zero three eight. The chi squared statistic measures how well the observed letter frequencies match expected english frequencies. Shannon entropy quantifies the information content per character. Autocorrelation at various lags can reveal periodic structure in the ciphertext which is characteristic of polyalphabetic ciphers. Bigram and trigram analysis helps distinguish between substitution and transposition ciphers since transposition ciphers preserve natural language bigram frequencies while substitution ciphers destroy them completely.

The quick brown fox jumps over the lazy dog this sentence contains every letter of the english alphabet and has been used since the late nineteenth century to test typewriters and computer keyboards a similar pangram used in typing practice is the five boxing wizards jump quickly both sentences provide excellent material for testing cryptographic implementations because they contain a complete set of alphabetic characters ensuring that every possible substitution mapping is exercised during encryption several other pangrams exist including pack my box with five dozen liquor jugs which is shorter at thirty two characters and how vexingly quick daft zebras jump which tests less common letter combinations
"""

    @staticmethod
    def generate_training_data(n_samples=10000):
        """
        Generate high-quality training data using the actual cipher implementations
        and a large diverse English corpus. Produces balanced classes with varied
        text lengths and random keys.
        """
        from ..ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair
        from ..utils.text_utils import clean_text
        
        cleaned_corpus = clean_text(CipherClassifier._CORPUS)
        corpus_length = len(cleaned_corpus)
        
        def get_random_plaintext(min_len=50, max_len=400):
            """Extract a random substring from the corpus with varied lengths."""
            target_len = random.randint(min_len, max_len)
            if corpus_length <= target_len:
                return cleaned_corpus
            start = random.randint(0, corpus_length - target_len)
            return cleaned_corpus[start:start + target_len]

        X, y = [], []
        extractor = FeatureExtractor()
        valid_a = affine.get_valid_a_values()
        samples_per_class = n_samples // len(CipherClassifier.CIPHER_TYPES)
        
        for ctype in CipherClassifier.CIPHER_TYPES:
            for i in range(samples_per_class):
                # Vary text length heavily towards short texts (what users typically test with)
                if i % 4 == 0:
                    pt = get_random_plaintext(20, 50)     # VERY short
                elif i % 4 == 1:
                    pt = get_random_plaintext(50, 100)    # short
                elif i % 4 == 2:
                    pt = get_random_plaintext(100, 200)   # medium
                else:
                    pt = get_random_plaintext(200, 400)   # long
                
                try:
                    if ctype == 'caesar':
                        shift = random.randint(1, 25)
                        ct = caesar.encrypt(pt, shift)
                    elif ctype == 'affine':
                        a = random.choice([x for x in valid_a if x != 1])  # exclude identity
                        b = random.randint(1, 25)
                        ct = affine.encrypt(pt, (a, b))
                    elif ctype == 'vigenere':
                        klen = random.randint(3, 8)
                        key = ''.join(random.choices(string.ascii_uppercase, k=klen))
                        ct = vigenere.encrypt(pt, key)
                    elif ctype == 'substitution':
                        key = substitution.generate_random_key()
                        ct = substitution.encrypt(pt, key)
                    elif ctype == 'columnar_transposition':
                        klen = random.randint(3, 8)
                        key = ''.join(random.choices(string.ascii_uppercase, k=klen))
                        ct = columnar_transposition.encrypt(pt, key)
                    elif ctype == 'playfair':
                        klen = random.randint(4, 10)
                        key = ''.join(random.choices(
                            string.ascii_uppercase.replace('J', ''), k=klen))
                        ct = playfair.encrypt(pt, key)
                    else:
                        ct = pt
                    
                    features = extractor.extract(ct)
                    X.append(features)
                    y.append(ctype)
                except Exception:
                    # Skip any samples that fail encryption (edge cases)
                    pass
                
        return np.array(X), np.array(y)
