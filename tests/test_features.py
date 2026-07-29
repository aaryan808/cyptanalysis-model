import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_feature_vector_length(self):
        vec = self.extractor.extract("TESTINGSTRING")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec), len(self.extractor.get_feature_names()))
        
        features = self.extractor.extract_dict("TESTINGSTRING")
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 50)  # 60+ features now
        self.assertIn('ic', features)
        self.assertIn('entropy', features)
        self.assertIn('ic_deviation', features)
        self.assertIn('sorted_freq_corr', features)

    def test_ic_english_text(self):
        text = "THISISASAMPLEENGLISHTEXTTHATSHOULDHAVEANINDEXOFCOINCIDENCEAROUNDTHENORMALVALUEFORSTANDARDENGLISHWHICHISSOMETHINGLIKESIXPOINTFIVETOSEVENPERCENT"
        features = self.extractor.extract_dict(text)
        ic = features.get('ic', 0)
        self.assertTrue(0.055 <= ic <= 0.080, f"Expected IC in [0.055, 0.080], got {ic}")

    def test_ic_random_text(self):
        import random
        random.seed(42)
        text = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(200))
        features = self.extractor.extract_dict(text)
        ic = features.get('ic', 0)
        self.assertTrue(0.030 <= ic <= 0.050, f"Expected IC in [0.030, 0.050], got {ic}")

    def test_entropy(self):
        text = "AAAAAAAAAAAAAAAA"
        features = self.extractor.extract_dict(text)
        entropy = features.get('entropy', 0)
        self.assertEqual(entropy, 0.0)

        text2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        features2 = self.extractor.extract_dict(text2)
        entropy2 = features2.get('entropy', 0)
        self.assertGreater(entropy2, 4.0)

    def test_chi_squared(self):
        english_text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGFREQUENTLYUSEDENGLISHWORDS"
        random_text = "XZYWQKVJBPOSUDMCGNLHARXFVTYZQ"
        
        feat_eng = self.extractor.extract_dict(english_text)
        feat_rnd = self.extractor.extract_dict(random_text)
        
        if 'chi_sq_eng' in feat_eng and 'chi_sq_eng' in feat_rnd:
            self.assertLess(feat_eng['chi_sq_eng'], feat_rnd['chi_sq_eng'])

    def test_sorted_freq_correlation(self):
        """Monoalphabetic substitution should preserve sorted frequency shape."""
        # Simple shift (Caesar) should have high correlation
        english = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 3
        shifted = "".join(chr(((ord(c) - 65 + 5) % 26) + 65) for c in english)
        feat = self.extractor.extract_dict(shifted)
        corr = feat.get('sorted_freq_corr', 0)
        self.assertGreater(corr, 0.7, f"Expected high correlation for Caesar, got {corr}")

if __name__ == '__main__':
    unittest.main()
