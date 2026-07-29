"""
Feature Extractor for Cipher Classification.

Extracts 60+ statistical features from ciphertext to enable high-accuracy
ML-based cipher type identification. Features are specifically designed to
discriminate between:
  - Monoalphabetic substitution (Caesar, Affine, general Substitution)
  - Polyalphabetic substitution (Vigenère)
  - Transposition ciphers (Columnar Transposition)
  - Digraphic ciphers (Playfair)
"""

import numpy as np
import math
from collections import Counter
import re

ENGLISH_FREQ = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}

ENGLISH_BIGRAMS = {
    'TH': 0.0356, 'HE': 0.0307, 'IN': 0.0243, 'ER': 0.0205, 'AN': 0.0199,
    'RE': 0.0185, 'ON': 0.0176, 'AT': 0.0149, 'EN': 0.0145, 'ND': 0.0135
}

# English IC ≈ 0.0667, random IC ≈ 1/26 ≈ 0.0385
ENGLISH_IC = 0.0667
RANDOM_IC = 1.0 / 26.0


class FeatureExtractor:
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
    def clean_text(self, text: str) -> str:
        """Clean input text to uppercase A-Z only."""
        return re.sub(r'[^A-Z]', '', text.upper())

    def _index_of_coincidence(self, text: str) -> float:
        """Calculate the Index of Coincidence (IC)."""
        n = len(text)
        if n < 2:
            return 0.0
        counts = Counter(text)
        ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
        return ic

    def _periodic_ic(self, text: str, period: int) -> float:
        """Calculate average IC across columns when text is arranged in `period` columns.
        High average IC at the correct period indicates polyalphabetic cipher key length."""
        if period < 1 or len(text) < period * 2:
            return 0.0
        columns = ['' for _ in range(period)]
        for i, c in enumerate(text):
            columns[i % period] += c
        ics = [self._index_of_coincidence(col) for col in columns if len(col) > 1]
        return np.mean(ics) if ics else 0.0

    def _best_periodic_ic(self, text: str, max_period: int = 15) -> tuple:
        """Find the period (2..max_period) with the highest average IC.
        Returns (best_period, best_ic, ic_ratio_to_random).
        For Vigenère, the best IC should be near English IC."""
        best_ic = 0.0
        best_period = 1
        for p in range(2, min(max_period + 1, len(text) // 2 + 1)):
            pic = self._periodic_ic(text, p)
            if pic > best_ic:
                best_ic = pic
                best_period = p
        ic_ratio = best_ic / ENGLISH_IC if ENGLISH_IC > 0 else 0.0
        return best_period, best_ic, ic_ratio

    def _sorted_freq_correlation(self, text: str) -> float:
        """Compute correlation between sorted observed frequencies and sorted English frequencies.
        High correlation → monoalphabetic substitution (frequency distribution is preserved, just permuted).
        Low correlation → polyalphabetic or transposition."""
        n = len(text)
        if n == 0:
            return 0.0
        counts = Counter(text)
        observed = sorted([counts.get(c, 0) / n for c in self.alphabet], reverse=True)
        expected = sorted(ENGLISH_FREQ.values(), reverse=True)
        obs_arr = np.array(observed)
        exp_arr = np.array(expected)
        if np.std(obs_arr) < 1e-10 or np.std(exp_arr) < 1e-10:
            return 0.0
        corr = np.corrcoef(obs_arr, exp_arr)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    def _digraph_reversal_score(self, text: str) -> float:
        """Measure how many bigrams have their reverse also present.
        Playfair ciphers tend to have more reversed bigram pairs because of the rectangle rule."""
        if len(text) < 4:
            return 0.0
        bigrams = set()
        for i in range(0, len(text) - 1, 2):  # Non-overlapping digraphs (Playfair uses pairs)
            bigrams.add(text[i:i+2])
        if not bigrams:
            return 0.0
        reversals = sum(1 for bg in bigrams if bg[::-1] in bigrams and bg != bg[::-1])
        return reversals / len(bigrams)

    def _log_digraph_score(self, text: str) -> float:
        """Score based on the log-probability of the text's bigram distribution matching English.
        Uses KL-divergence-like measure."""
        if len(text) < 2:
            return 0.0
        bigrams = [text[i:i+2] for i in range(len(text) - 1)]
        total = len(bigrams)
        bigram_counts = Counter(bigrams)
        score = 0.0
        for bg, expected_freq in ENGLISH_BIGRAMS.items():
            observed_freq = bigram_counts.get(bg, 0) / total
            if observed_freq > 0:
                score += observed_freq * math.log(observed_freq / expected_freq)
            else:
                score += 0  # skip zero-observed
        return score

    def extract(self, text: str) -> np.ndarray:
        """Extract all features from ciphertext, returns feature vector."""
        cleaned = self.clean_text(text)
        length = len(cleaned)
        if length == 0:
            return np.zeros(len(self.get_feature_names()))

        # === 1. Letter frequencies (26 features) ===
        counts = Counter(cleaned)
        freqs = {char: counts[char] / length for char in self.alphabet}
        freq_list = [freqs[char] for char in self.alphabet]

        # === 2. Overall IC ===
        ic = self._index_of_coincidence(cleaned)

        # === 3. Shannon entropy ===
        entropy = -sum(f * math.log2(f) for f in freq_list if f > 0)

        # === 4. Chi-squared statistic against English ===
        chi_sq = sum(((freqs[char] - ENGLISH_FREQ[char]) ** 2) / ENGLISH_FREQ[char]
                     for char in self.alphabet)

        # === 5-7. Frequency distribution stats ===
        max_freq = max(freq_list)
        min_freq = min(freq_list)
        std_freq = float(np.std(freq_list))

        # === 8-9. Even/Odd IC (helps detect digraphic ciphers) ===
        even_ic = self._index_of_coincidence(cleaned[0::2])
        odd_ic = self._index_of_coincidence(cleaned[1::2])

        # === 10. Bigram repeat rate ===
        bigrams = [cleaned[i:i+2] for i in range(length - 1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for v in bigram_counts.values() if v > 1)
            bigram_repeat_rate = repeated_bigrams / len(bigram_counts)
        else:
            bigram_repeat_rate = 0.0

        # === 11-12. Double letter features ===
        double_letters_count = sum(1 for i in range(length - 1)
                                   if cleaned[i] == cleaned[i+1])
        has_double = 1.0 if double_letters_count > 0 else 0.0
        double_ratio = double_letters_count / length if length > 0 else 0.0

        # === 13. Text length parity ===
        length_parity = float(length % 2)

        # === 14. Autocorrelation at lags 1-5 ===
        autocorrs = []
        for lag in range(1, 6):
            if length > lag:
                lag_matches = sum(1 for i in range(length - lag)
                                 if cleaned[i] == cleaned[i+lag])
                autocorrs.append(lag_matches / (length - lag))
            else:
                autocorrs.append(0.0)

        # === 15. Pattern repetition score (Kasiski-like) ===
        trigrams = [cleaned[i:i+3] for i in range(length - 2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(v for v in trigram_counts.values() if v > 1)
            pattern_rep_score = repeated_trigrams / len(trigrams)
        else:
            pattern_rep_score = 0.0

        # === 16. Digram chi-squared ===
        digram_freqs = {bg: 0.0 for bg in ENGLISH_BIGRAMS}
        if bigrams:
            total_bgs = len(bigrams)
            for bg in ENGLISH_BIGRAMS:
                digram_freqs[bg] = bigram_counts.get(bg, 0) / total_bgs
            digram_chi_sq = sum(((digram_freqs[bg] - ENGLISH_BIGRAMS[bg]) ** 2) / ENGLISH_BIGRAMS[bg]
                                for bg in ENGLISH_BIGRAMS)
        else:
            digram_chi_sq = 0.0

        # ============================================================
        # NEW DISCRIMINATIVE FEATURES (17-30+)
        # ============================================================

        # === 17. IC deviation from English ===
        ic_deviation = abs(ic - ENGLISH_IC)

        # === 18. IC ratio (IC / English_IC): >0.9 = monoalpha or transposition ===
        ic_ratio = ic / ENGLISH_IC if ENGLISH_IC > 0 else 0.0

        # === 19-21. Best periodic IC (Vigenère key-length detector) ===
        best_period, best_periodic_ic, periodic_ic_ratio = self._best_periodic_ic(cleaned)

        # === 22. Periodic IC jump: how much does best periodic IC exceed overall IC ===
        periodic_ic_jump = best_periodic_ic - ic

        # === 23. Sorted frequency correlation with English ===
        sorted_freq_corr = self._sorted_freq_correlation(cleaned)

        # === 24. Number of unique letters used ===
        unique_letters = len(counts)
        unique_letter_ratio = unique_letters / 26.0

        # === 25. Frequency of the most common letter (normalized to English 'E') ===
        top1_ratio = max_freq / 0.12702 if 0.12702 > 0 else 0.0

        # === 26. Frequency range (max - min) ===
        freq_range = max_freq - min_freq

        # === 27. Kurtosis of letter frequency distribution ===
        freq_arr = np.array(freq_list)
        mu = np.mean(freq_arr)
        sigma = np.std(freq_arr)
        if sigma > 1e-10:
            kurtosis = float(np.mean(((freq_arr - mu) / sigma) ** 4) - 3.0)
        else:
            kurtosis = 0.0

        # === 28. Skewness of letter frequency distribution ===
        if sigma > 1e-10:
            skewness = float(np.mean(((freq_arr - mu) / sigma) ** 3))
        else:
            skewness = 0.0

        # === 29. Digraph reversal score (Playfair indicator) ===
        digraph_reversal = self._digraph_reversal_score(cleaned)

        # === 30. Even IC / Odd IC ratio (near 1.0 for most ciphers,
        #     differs for transposition) ===
        even_odd_ic_ratio = even_ic / odd_ic if odd_ic > 1e-10 else 1.0

        # === 31. Log digraph divergence ===
        log_digraph = self._log_digraph_score(cleaned)

        # === 32. Text length (normalized, log scale) ===
        log_length = math.log(length) if length > 0 else 0.0

        # === 33. Chi-squared per letter (normalized chi-sq) ===
        chi_sq_per_letter = chi_sq / 26.0

        # Assemble full feature vector
        features = freq_list + [
            ic, entropy, chi_sq, max_freq, min_freq, std_freq,
            even_ic, odd_ic, bigram_repeat_rate, has_double,
            double_ratio, length_parity
        ] + autocorrs + [
            pattern_rep_score, digram_chi_sq,
            # Discriminative features
            ic_deviation, ic_ratio,
            best_periodic_ic, periodic_ic_ratio, periodic_ic_jump,
            float(best_period),
            sorted_freq_corr,
            unique_letter_ratio, top1_ratio, freq_range,
            kurtosis, skewness,
            digraph_reversal, even_odd_ic_ratio,
            log_digraph, log_length, chi_sq_per_letter
        ]

        return np.array(features, dtype=float)

    def extract_dict(self, text: str) -> dict:
        """Extract all features from ciphertext, returns dictionary mapping feature name to value."""
        feature_vec = self.extract(text)
        names = self.get_feature_names()
        return {name: round(float(val), 6) for name, val in zip(names, feature_vec)}

    def get_feature_names(self) -> list:
        """Return names of all features."""
        names = [f"freq_{char}" for char in self.alphabet]
        names += [
            "ic", "entropy", "chi_sq_eng", "max_freq", "min_freq", "std_freq",
            "even_ic", "odd_ic", "bigram_repeat_rate", "has_double_letters",
            "double_letter_ratio", "length_parity",
            "autocorr_lag_1", "autocorr_lag_2", "autocorr_lag_3", "autocorr_lag_4", "autocorr_lag_5",
            "pattern_rep_score", "digram_chi_sq",
            # New discriminative features
            "ic_deviation", "ic_ratio",
            "best_periodic_ic", "periodic_ic_ratio", "periodic_ic_jump",
            "best_period",
            "sorted_freq_corr",
            "unique_letter_ratio", "top1_ratio", "freq_range",
            "kurtosis", "skewness",
            "digraph_reversal", "even_odd_ic_ratio",
            "log_digraph", "log_length", "chi_sq_per_letter"
        ]
        return names
