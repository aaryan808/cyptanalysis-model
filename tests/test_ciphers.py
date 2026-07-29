import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair

class TestCaesar(unittest.TestCase):
    def test_encrypt_decrypt(self):
        plaintext = "HELLOCAESAR"
        key = 3
        ciphertext = caesar.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = caesar.decrypt(ciphertext, key)
        self.assertEqual(decrypted, plaintext)

    def test_edge_cases(self):
        self.assertEqual(caesar.encrypt("", 5), "")
        self.assertEqual(caesar.encrypt("A", 26), "A")
        self.assertEqual(caesar.encrypt("Z", 1), "A")
        self.assertEqual(caesar.decrypt("A", 1), "Z")

class TestAffine(unittest.TestCase):
    def test_encrypt_decrypt(self):
        plaintext = "AFFINECIPHER"
        key = (5, 8)
        ciphertext = affine.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = affine.decrypt(ciphertext, key)
        self.assertEqual(decrypted, plaintext)

    def test_invalid_key(self):
        with self.assertRaises(Exception):
            affine.encrypt("TEST", (4, 7)) # gcd(4, 26) != 1

    def test_edge_cases(self):
        self.assertEqual(affine.encrypt("", (5, 8)), "")
        self.assertEqual(affine.encrypt("A", (1, 0)), "A")

class TestVigenere(unittest.TestCase):
    def test_encrypt_decrypt(self):
        plaintext = "VIGENERECIPHER"
        key = "KEY"
        ciphertext = vigenere.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = vigenere.decrypt(ciphertext, key)
        self.assertEqual(decrypted, plaintext)

    def test_edge_cases(self):
        self.assertEqual(vigenere.encrypt("", "KEY"), "")
        self.assertEqual(vigenere.encrypt("A", "A"), "A")

class TestSubstitution(unittest.TestCase):
    def test_encrypt_decrypt(self):
        plaintext = "SUBSTITUTION"
        key = "QWERTYUIOPASDFGHJKLZXCVBNM"
        ciphertext = substitution.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = substitution.decrypt(ciphertext, key)
        self.assertEqual(decrypted, plaintext)

    def test_edge_cases(self):
        self.assertEqual(substitution.encrypt("", "QWERTYUIOPASDFGHJKLZXCVBNM"), "")

class TestColumnarTransposition(unittest.TestCase):
    def test_encrypt_decrypt_word_key(self):
        plaintext = "TRANSPOSITION"
        key = "KEY"
        ciphertext = columnar_transposition.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = columnar_transposition.decrypt(ciphertext, key)
        self.assertTrue(decrypted.startswith(plaintext))

    def test_edge_cases(self):
        self.assertEqual(columnar_transposition.encrypt("", "KEY"), "")

class TestPlayfair(unittest.TestCase):
    def test_encrypt_decrypt(self):
        plaintext = "PLAYFAIR"
        key = "MONARCHY"
        ciphertext = playfair.encrypt(plaintext, key)
        self.assertNotEqual(ciphertext, plaintext)
        decrypted = playfair.decrypt(ciphertext, key)
        self.assertTrue(decrypted.startswith("PLAYFAIR") or decrypted.startswith("PLAIFAIR"))

    def test_edge_cases(self):
        self.assertEqual(playfair.encrypt("", "KEY"), "")

if __name__ == '__main__':
    unittest.main()
