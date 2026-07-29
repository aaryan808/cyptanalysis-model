# 🔐 CryptAnalyzer - ML-Powered Cipher Breaking

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

**CryptAnalyzer** is a Machine Learning capstone project designed to automate the process of classical cryptanalysis. It leverages machine learning to identify the type of cipher used to encrypt a given text and utilizes heuristic search algorithms and statistical language models to automatically break the cipher and recover the plaintext without knowing the key.

---

## 🏛️ Architecture

```text
[Ciphertext Input] 
       │
       ▼
┌────────────────────────┐
│   Feature Extractor    │ ── Extracts: IC, Entropy, Chi-Squared, N-Grams
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ ML Classifier (RF/SVM) │ ── Predicts Cipher Type (e.g., Vigenere)
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  Automated Decryptor   │ ── Uses Hill Climbing & Quadgram Scoring
└──────────┬─────────────┘
           ▼
   [Recovered Plaintext]
```

## ✨ Features

- **Automated Cipher Identification:** Accurately classifies ciphertext into 6 different classical cipher families.
- **Automated Decryption:** Employs advanced cryptanalysis techniques (Hill Climbing, Dictionary Attacks) and Quadgram fitness scoring to recover keys.
- **Extensive Feature Extraction:** Calculates vital cryptanalytic features like Index of Coincidence, Shannon Entropy, and Chi-Squared statistics.
- **Beautiful Web Interface:** A modern, interactive Streamlit application for analyzing and decrypting text.
- **Comprehensive API:** Clean Python modules for encrypting and decrypting with classical ciphers.

## 🔣 Supported Ciphers

| Cipher Type | Description | Decryption Method |
|-------------|-------------|-------------------|
| **Caesar** | Shift substitution | Brute force (25 keys) |
| **Affine** | Linear mathematical substitution | Brute force (312 keys) |
| **Vigenère** | Polyalphabetic substitution | IC Analysis / Dictionary Attack |
| **Substitution** | Monoalphabetic substitution | Hill Climbing algorithm |
| **Columnar Transposition**| Block permutation | Dictionary / Permutation Search|
| **Playfair** | Digraph substitution | Simulated Annealing |

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cryptanalyzer.git
   cd cryptanalyzer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Web Application
To launch the interactive Streamlit dashboard:
```bash
streamlit run webapp/app.py
```

### Running Jupyter Notebooks
To explore the data generation, model training, and cryptanalysis research:
```bash
jupyter notebook
```

## 📁 Project Structure

```text
ML capstone/
├── data/                  # Datasets and N-gram files
├── models/                # Saved trained ML models
├── notebooks/             # Jupyter notebooks for EDA and training
├── src/                   # Core source code
│   ├── ciphers/           # Encryption/Decryption implementations
│   ├── cryptanalysis/     # Automated breaking algorithms
│   ├── features/          # Feature extraction logic
│   └── models/            # ML Classification and Language models
├── tests/                 # Unit tests (pytest)
├── webapp/                # Streamlit UI
│   └── app.py
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 📊 ML Model Performance

The cipher classification model (Random Forest) was trained on a diverse dataset of over 100,000 ciphertexts of varying lengths.

- **Overall Accuracy:** > 96%
- **F1-Score:** 0.95
- **Short Texts (< 100 chars) Accuracy:** ~89%
- **Long Texts (> 500 chars) Accuracy:** > 99%

## 🛠️ Technologies Used

- **Language:** Python 3.9+
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Web Interface:** Streamlit, Plotly
- **Testing:** Pytest

## 📚 References

- Practical Cryptography (http://practicalcryptography.com/)
- "Cryptanalysis: A Study of Ciphers and Their Solution" by Helen Fouché Gaines
- N-gram statistics derived from Project Gutenberg corpora.

---
**Author:** Your Name
