# 🎬 CineClassify — Movie Genre Prediction with NLP & ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20LSA-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A machine learning pipeline that predicts movie genres from plot summaries using TF-IDF vectorization, Latent Semantic Analysis, and multiple classical classifiers.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [Models](#-models--results) · [Architecture](#-architecture) · [Contributing](#-contributing)

</div>

---

## 📌 Overview

CineClassify is an end-to-end NLP classification system that takes a movie's plot summary as input and predicts its genre using trained machine learning models. The project demonstrates the complete ML workflow — from raw text preprocessing and feature engineering to model training, evaluation, and an interactive web-based prediction interface.

### Supported Genres

| Icon | Genre | Icon | Genre | Icon | Genre |
|------|-------|------|-------|------|-------|
| 💥 | Action | 🎨 | Animation | 😂 | Comedy |
| 📽 | Documentary | 🎭 | Drama | 👻 | Horror |
| 💖 | Romance | 🚀 | Science Fiction | 🔪 | Thriller |

---

## ✨ Features

- **Multi-classifier comparison** — Complement Naive Bayes, Logistic Regression, Linear SVM, and LSA + Logistic Regression evaluated side-by-side
- **TF-IDF feature engineering** — unigram + bigram tokenization with sublinear TF scaling across 15,000 features
- **Latent Semantic Analysis** — TruncatedSVD reduces TF-IDF vectors to 150-dimensional word-embedding proxies
- **Robust evaluation** — 5-fold stratified cross-validation with accuracy, weighted F1, and per-class metrics
- **Interactive dashboard** — browser-based HTML app with live genre prediction, probability bars, and keyword highlighting
- **Rich visualizations** — confusion matrix, per-genre F1 chart, model comparison bars, and dataset distribution plot

---

## 🖥 Demo

Open `movie_genre_predictor.html` in any modern browser — no server required.

```
Input:  "A retired hitman is pulled back into the underworld when his
         former employer kidnaps his daughter..."

Output: 💥 Action  (confidence: 94.2%)
```

**Keyboard shortcut:** `Ctrl + Enter` to predict from the text area.

---

## 📁 Project Structure

```
cine-classify/
│
├── train_model.py                  # Full ML training pipeline
├── movie_genre_predictor.html      # Interactive browser demo
│
├── outputs/
│   ├── best_model.pkl              # Serialized best classifier (joblib)
│   ├── label_encoder.pkl           # Genre label encoder
│   ├── dataset.csv                 # Generated training dataset
│   └── results.json                # Evaluation metrics (JSON)
│
├── figures/
│   ├── fig_model_comparison.png    # Accuracy / F1 / CV bar charts
│   ├── fig_confusion_matrix.png    # Normalized confusion matrix
│   ├── fig_per_genre_f1.png        # Per-genre F1 scores
│   └── fig_dataset_dist.png        # Sample distribution by genre
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cine-classify.git
cd cine-classify

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Train the models

```bash
python train_model.py
```

This will:
1. Generate and augment the training dataset (703 samples, 9 genres)
2. Train four classifiers with TF-IDF features
3. Run 5-fold stratified cross-validation
4. Save the best model to `outputs/best_model.pkl`
5. Export four performance visualization figures

### Run the interactive demo

```bash
# Simply open in a browser — no server needed
open movie_genre_predictor.html          # macOS
start movie_genre_predictor.html         # Windows
xdg-open movie_genre_predictor.html      # Linux
```

### Use the trained model programmatically

```python
import joblib

# Load artifacts
model = joblib.load("outputs/best_model.pkl")
le    = joblib.load("outputs/label_encoder.pkl")

# Predict a single plot summary
plot = """
A forensic psychologist is drawn into a deadly cat-and-mouse game
with a calculating serial killer who seems to know her every move.
"""

prediction = model.predict([plot])
genre = le.inverse_transform(prediction)[0]
print(f"Predicted genre: {genre}")   # → Thriller
```

---

## 📊 Models & Results

All four classifiers achieved perfect scores on this dataset due to its strongly genre-distinctive vocabulary. In production settings with noisier, real-world data, scores will vary.

| Model | Test Accuracy | Weighted F1 | CV Mean ± Std |
|---|---|---|---|
| ⭐ Complement Naive Bayes | 100.0% | 1.000 | 1.000 ± 0.000 |
| Logistic Regression | 100.0% | 1.000 | 1.000 ± 0.000 |
| Linear SVM | 100.0% | 1.000 | 1.000 ± 0.000 |
| LSA + Logistic Regression | 100.0% | 1.000 | 1.000 ± 0.000 |

> **Best model selected:** Complement Naive Bayes — preferred for text classification due to its strong performance on imbalanced corpora and interpretable probabilistic output.

---

## 🏗 Architecture

```
Raw Plot Text
      │
      ▼
Tokenize & Lowercase
      │
      ▼
Remove Stopwords
      │
      ▼
TF-IDF Vectorizer ─── (1,2)-grams · 15K features · sublinear TF
      │
      ├──► Complement Naive Bayes    ──► Genre Prediction
      ├──► Logistic Regression       ──► Genre Prediction
      ├──► Linear SVM                ──► Genre Prediction
      └──► TruncatedSVD (150d) + LR  ──► Genre Prediction (LSA variant)
```

### Why TF-IDF + Bigrams?

- **Term Frequency-Inverse Document Frequency** down-weights common words and amplifies discriminative terms
- **Bigrams** (`(1,2)-gram range`) capture two-word phrases like `"serial killer"`, `"falls in love"`, or `"space colony"` that carry strong genre signals
- **Sublinear TF** (`log(1 + tf)`) prevents high-frequency terms from dominating the feature space

### Why Complement Naive Bayes?

Standard Multinomial Naive Bayes estimates `P(word | class)`. Complement NB instead models `P(word | NOT class)` and has been shown to outperform the standard variant on text classification tasks, particularly where class sizes are not perfectly balanced.

---

## 📈 Visualizations

| Figure | Description |
|--------|-------------|
| `fig_model_comparison.png` | Side-by-side bar charts comparing accuracy, F1, and cross-val scores across all four models |
| `fig_confusion_matrix.png` | Normalized heatmap showing true vs. predicted genre distributions |
| `fig_per_genre_f1.png` | Per-genre F1 scores with mean baseline reference line |
| `fig_dataset_dist.png` | Sample count distribution across the 9 genre classes |

---

## 🗂 Dataset

The training corpus was synthetically generated to demonstrate the pipeline. Each genre contains ~80 plot summaries, augmented by cross-sentence recombination for variety.

| Property | Value |
|---|---|
| Total samples | 703 |
| Genres | 9 |
| Train / Test split | 80% / 20% (stratified) |
| Augmentation | Sentence-level recombination |

To use a real-world dataset (e.g. [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) or [IMDb](https://www.imdb.com/interfaces/)), replace the `generate_dataset()` function in `train_model.py` with a CSV loader:

```python
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["plot", "genre"])
    return df.dropna().reset_index(drop=True)
```

---

## 🔧 Extending the Project

**Add a new classifier:**

```python
from sklearn.ensemble import GradientBoostingClassifier

models["Gradient Boosting"] = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("clf",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
])
```

**Swap TF-IDF for word embeddings:**

```python
# Using spaCy sentence vectors
import spacy
nlp = spacy.load("en_core_web_md")

X_vecs = [nlp(text).vector for text in df["plot"]]
```

**Enable multi-label classification** (a film can be both Action and Thriller):

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["genres"])   # genres is a list of lists

clf = OneVsRestClassifier(LinearSVC())
clf.fit(X_train, Y_train)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/add-bert-embeddings`
3. Commit your changes: `git commit -m "feat: add BERT sentence embeddings"`
4. Push to the branch: `git push origin feature/add-bert-embeddings`
5. Open a Pull Request

Please make sure your code follows [PEP 8](https://peps.python.org/pep-0008/) and includes appropriate docstrings.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — machine learning library
- [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) — inspiration for dataset structure
- [Rennie et al. (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) — original Complement Naive Bayes paper

---

<div align="center">

Made with ❤️ and Python · Star ⭐ this repo if you found it useful

</div>
