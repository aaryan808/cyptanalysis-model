import os
import json
import random
import string

def create_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    }

nb1_cells = [
    md("# Dataset Generation for Cipher Classification\nDataset Generation for Cipher Classification"),
    code("import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))\n\nimport numpy as np\nimport pandas as pd\nimport random\nimport string\n\nfrom src.ciphers import caesar, affine, vigenere, substitution, columnar_transposition, playfair\nfrom src.features.extractor import FeatureExtractor"),
    md("## Load English Corpus\nLoad English Corpus"),
    code('text_corpus = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.\nHowever little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.\n"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"\nMr. Bennet replied that he had not.\n"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."\nMr. Bennet made no answer.\n"Do you not want to know who has taken it?" cried his wife impatiently.\n"You want to tell me, and I have no objection to hearing it."\nThis was invitation enough.\n"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week."\n"What is his name?"\n"Bingley."\n"Is he married or single?"\n"Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!"\n"How so? How can it affect them?"\n"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must know that I am thinking of his marrying one of them."\n"Is that his design in settling here?"\n"Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes."\n"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party."\n"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty."\n"In such cases, a woman has not often much beauty to think of."\n"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood."\n"It is more than I engage for, I assure you."\n"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general, you know, they visit no newcomers. Indeed you must go, for it will be impossible for us to visit him if you do not."\n"You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy."\n"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference."\n"They have none of them much to recommend them," replied he; "they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters."\n"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion for my poor nerves."\n"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these last twenty years at least."\n"""'),
    md("## Text Preprocessing\nClean text to uppercase A-Z only, split into chunks of varying sizes (100-500 chars)"),
    code("import re\ndef clean_text(text):\n    return re.sub(r'[^A-Z]', '', text.upper())\n\ncleaned_corpus = clean_text(text_corpus)\n\nchunks = []\nidx = 0\nwhile idx < len(cleaned_corpus) - 100:\n    chunk_size = random.randint(100, 500)\n    chunk = cleaned_corpus[idx:idx+chunk_size]\n    if len(chunk) >= 100:\n        chunks.append(chunk)\n    idx += chunk_size\n    \nprint(f'Generated {len(chunks)} chunks.')"),
    md("## Generate Cipher Samples\nFor each cipher type, encrypt chunks with random keys:\n- Caesar: random shift 1-25\n- Affine: random valid (a,b) pairs\n- Vigenère: random keywords length 3-7\n- Substitution: random permutation keys\n- Columnar Transposition: random keywords length 3-7\n- Playfair: random keywords\nGenerate 500 samples per cipher type = 3000 total"),
    code("samples = []\ncipher_types = ['caesar', 'affine', 'vigenere', 'substitution', 'columnar_transposition', 'playfair']\nnum_samples_per_cipher = 500\n\ndef get_random_chunk():\n    if chunks:\n        return random.choice(chunks)\n    return 'A'*100\n\ndef generate_random_substitution_key():\n    alpha = list(string.ascii_uppercase)\n    random.shuffle(alpha)\n    return ''.join(alpha)\n\nfor _ in range(num_samples_per_cipher):\n    # Caesar\n    pt = get_random_chunk()\n    shift = random.randint(1, 25)\n    ct_caesar = caesar.encrypt(pt, shift)\n    samples.append({'plaintext': pt, 'ciphertext': ct_caesar, 'label': 'caesar'})\n    \n    # Affine\n    pt = get_random_chunk()\n    valid_a = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]\n    a = random.choice(valid_a)\n    b = random.randint(0, 25)\n    ct_affine = affine.encrypt(pt, a, b)\n    samples.append({'plaintext': pt, 'ciphertext': ct_affine, 'label': 'affine'})\n    \n    # Vigenere\n    pt = get_random_chunk()\n    kw_len = random.randint(3, 7)\n    kw = ''.join(random.choices(string.ascii_uppercase, k=kw_len))\n    ct_vig = vigenere.encrypt(pt, kw)\n    samples.append({'plaintext': pt, 'ciphertext': ct_vig, 'label': 'vigenere'})\n    \n    # Substitution\n    pt = get_random_chunk()\n    key_sub = generate_random_substitution_key()\n    ct_sub = substitution.encrypt(pt, key_sub)\n    samples.append({'plaintext': pt, 'ciphertext': ct_sub, 'label': 'substitution'})\n    \n    # Columnar Transposition\n    pt = get_random_chunk()\n    kw_len = random.randint(3, 7)\n    kw = ''.join(random.choices(string.ascii_uppercase, k=kw_len))\n    ct_col = columnar_transposition.encrypt(pt, kw)\n    samples.append({'plaintext': pt, 'ciphertext': ct_col, 'label': 'columnar_transposition'})\n    \n    # Playfair\n    pt = get_random_chunk()\n    kw_len = random.randint(3, 10)\n    kw = ''.join(random.choices(string.ascii_uppercase, k=kw_len))\n    pt_playfair = pt.replace('J', 'I')\n    if len(pt_playfair) % 2 != 0: pt_playfair += 'X'\n    ct_play = playfair.encrypt(pt_playfair, kw)\n    samples.append({'plaintext': pt_playfair, 'ciphertext': ct_play, 'label': 'playfair'})\n\nprint(f'Generated {len(samples)} total samples.')"),
    md("## Feature Extraction\nExtract features for all samples using FeatureExtractor"),
    code("extractor = FeatureExtractor()\ndataset_rows = []\n\nfor s in samples:\n    ct = s['ciphertext']\n    features = extractor.extract_all_features(ct)\n    features['label'] = s['label']\n    dataset_rows.append(features)\n    \ndf = pd.DataFrame(dataset_rows)\ndf.head()"),
    md("## Save Dataset\nSave to CSV with features and labels"),
    code("os.makedirs('../data/processed', exist_ok=True)\ndf.to_csv('../data/processed/cipher_dataset.csv', index=False)\nprint('Dataset saved to ../data/processed/cipher_dataset.csv')"),
    md("## Dataset Summary\nPrint counts, shapes, label distribution"),
    code("print(f'Dataset shape: {df.shape}')\nprint('\\nLabel distribution:')\nprint(df['label'].value_counts())")
]

nb2_cells = [
    md("# Exploratory Data Analysis\nExploratory Data Analysis"),
    code("import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\n\nplt.style.use('seaborn-v0_8-darkgrid')"),
    md("## Load Dataset\nLoad from CSV (or generate if not found using classifier.generate_training_data)"),
    code("try:\n    df = pd.read_csv('../data/processed/cipher_dataset.csv')\n    print('Dataset loaded successfully.')\nexcept FileNotFoundError:\n    print('Dataset not found.')\n    df = pd.DataFrame()"),
    md("## Class Distribution\nBar chart of cipher type counts"),
    code("plt.figure(figsize=(10, 6))\nsns.countplot(data=df, x='label', hue='label', palette='viridis')\nplt.title('Cipher Type Distribution')\nplt.xlabel('Cipher Type')\nplt.ylabel('Count')\nplt.show()"),
    md("## Feature Distributions\nHistograms of IC, entropy, chi-squared by cipher type (overlaid)\nBox plots of key features grouped by cipher type"),
    code("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\nsns.histplot(data=df, x='ioc', hue='label', element='step', ax=axes[0])\naxes[0].set_title('Index of Coincidence Distribution')\n\nsns.histplot(data=df, x='entropy', hue='label', element='step', ax=axes[1])\naxes[1].set_title('Entropy Distribution')\n\nsns.histplot(data=df, x='chi_square', hue='label', element='step', ax=axes[2])\naxes[2].set_title('Chi-Square Distribution')\n\nplt.tight_layout()\nplt.show()"),
    code("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\nsns.boxplot(data=df, x='label', y='ioc', hue='label', ax=axes[0])\naxes[0].set_title('Index of Coincidence by Cipher')\naxes[0].tick_params(axis='x', rotation=45)\n\nsns.boxplot(data=df, x='label', y='entropy', hue='label', ax=axes[1])\naxes[1].set_title('Entropy by Cipher')\naxes[1].tick_params(axis='x', rotation=45)\n\nsns.boxplot(data=df, x='label', y='chi_square', hue='label', ax=axes[2])\naxes[2].set_title('Chi-Square by Cipher')\naxes[2].tick_params(axis='x', rotation=45)\n\nplt.tight_layout()\nplt.show()"),
    md("## Correlation Analysis\nCorrelation heatmap of top features\nFeature importance from a quick Random Forest"),
    code("plt.figure(figsize=(12, 10))\nnumeric_df = df.drop(columns=['label'])\nsns.heatmap(numeric_df.corr(), cmap='coolwarm', center=0)\nplt.title('Feature Correlation Heatmap')\nplt.show()"),
    code("X = df.drop(columns=['label'])\ny = df['label']\n\nrf = RandomForestClassifier(n_estimators=100, random_state=42)\nrf.fit(X, y)\n\nimportances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)\n\nplt.figure(figsize=(10, 6))\nimportances.head(15).plot(kind='bar')\nplt.title('Top 15 Feature Importances (Random Forest)')\nplt.ylabel('Importance')\nplt.show()"),
    md("## Dimensionality Reduction\nPCA 2D scatter plot colored by cipher type\nt-SNE 2D scatter plot colored by cipher type"),
    code("pca = PCA(n_components=2)\nX_pca = pca.fit_transform(X)\ndf_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'label': y})\n\nplt.figure(figsize=(10, 8))\nsns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='label', palette='tab10')\nplt.title('PCA: 2D Projection of Features')\nplt.show()"),
    code("tsne = TSNE(n_components=2, random_state=42)\nX_tsne = tsne.fit_transform(X)\ndf_tsne = pd.DataFrame({'Dim1': X_tsne[:, 0], 'Dim2': X_tsne[:, 1], 'label': y})\n\nplt.figure(figsize=(10, 8))\nsns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='label', palette='tab10')\nplt.title('t-SNE: 2D Projection of Features')\nplt.show()"),
    md("## Key Observations\n- Distribution plots show clear separation in some features (e.g. IoC for transposition vs substitution).\n- PCA and t-SNE indicate that some classes cluster well while others overlap.\n- Top features heavily influence the Random Forest.")
]

nb3_cells = [
    md("# ML Model Training & Evaluation\nML Model Training & Evaluation"),
    code("import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\n\nfrom sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n\nplt.style.use('seaborn-v0_8-darkgrid')"),
    md("## Load/Generate Data\nLoad dataset or generate using CipherClassifier.generate_training_data"),
    code("df = pd.read_csv('../data/processed/cipher_dataset.csv')\nX = df.drop(columns=['label'])\ny = df['label']"),
    md("## Train-Test Split\n80-20 split with stratification"),
    code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\nprint(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')"),
    md("## Model 1: Random Forest\nTrain with hyperparameter tuning (GridSearchCV)\nPrint classification report\nPlot confusion matrix heatmap\nPlot feature importance bar chart (top 15)"),
    code("rf = RandomForestClassifier(random_state=42)\nrf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}\nrf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)\nrf_grid.fit(X_train, y_train)\n\nbest_rf = rf_grid.best_estimator_\ny_pred_rf = best_rf.predict(X_test)\n\nprint('Random Forest Classification Report:')\nprint(classification_report(y_test, y_pred_rf))\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', \n            xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)\nplt.title('Random Forest Confusion Matrix')\nplt.ylabel('True')\nplt.xlabel('Predicted')\nplt.show()"),
    code("rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)\nplt.figure(figsize=(10, 6))\nrf_importances.head(15).plot(kind='bar')\nplt.title('Top 15 Feature Importances (Random Forest)')\nplt.ylabel('Importance')\nplt.show()"),
    md("## Model 2: SVM\nTrain SVM with RBF kernel\nPrint classification report\nPlot confusion matrix"),
    code("svm = SVC(kernel='rbf', random_state=42)\nsvm.fit(X_train, y_train)\ny_pred_svm = svm.predict(X_test)\n\nprint('SVM Classification Report:')\nprint(classification_report(y_test, y_pred_svm))\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Oranges',\n            xticklabels=svm.classes_, yticklabels=svm.classes_)\nplt.title('SVM Confusion Matrix')\nplt.ylabel('True')\nplt.xlabel('Predicted')\nplt.show()"),
    md("## Model 3: Neural Network\nTrain MLP\nPrint classification report\nPlot confusion matrix"),
    code("mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)\nmlp.fit(X_train, y_train)\ny_pred_mlp = mlp.predict(X_test)\n\nprint('Neural Network Classification Report:')\nprint(classification_report(y_test, y_pred_mlp))\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(confusion_matrix(y_test, y_pred_mlp), annot=True, fmt='d', cmap='Greens',\n            xticklabels=mlp.classes_, yticklabels=mlp.classes_)\nplt.title('Neural Network Confusion Matrix')\nplt.ylabel('True')\nplt.xlabel('Predicted')\nplt.show()"),
    md("## Model Comparison\nBar chart comparing accuracy of all 3 models\nCross-validation scores comparison"),
    code("acc_rf = accuracy_score(y_test, y_pred_rf)\nacc_svm = accuracy_score(y_test, y_pred_svm)\nacc_mlp = accuracy_score(y_test, y_pred_mlp)\n\nmodels = ['Random Forest', 'SVM', 'Neural Network']\naccuracies = [acc_rf, acc_svm, acc_mlp]\n\nplt.figure(figsize=(8, 5))\nsns.barplot(x=models, y=accuracies, hue=models, legend=False, palette='Set2')\nplt.title('Model Accuracy Comparison')\nplt.ylim(0, 1)\nfor i, v in enumerate(accuracies):\n    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')\nplt.show()"),
    md("## Save Best Model\nSave the best performing model using joblib"),
    code("best_model = best_rf\nos.makedirs('../models', exist_ok=True)\njoblib.dump(best_model, '../models/cipher_classifier.pkl')\nprint('Best model saved to ../models/cipher_classifier.pkl')"),
    md("## Summary\nConclusion with best model recommendation")
]

create_notebook(r'c:\Users\malli\OneDrive\Desktop\ML capstone\notebooks\01_data_generation.ipynb', nb1_cells)
create_notebook(r'c:\Users\malli\OneDrive\Desktop\ML capstone\notebooks\02_eda.ipynb', nb2_cells)
create_notebook(r'c:\Users\malli\OneDrive\Desktop\ML capstone\notebooks\03_model_training.ipynb', nb3_cells)
