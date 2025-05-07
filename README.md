
# Twitter Data Sentiment Analysis

This project performs sentiment analysis on Twitter data using advanced NLP techniques and deep learning. The analysis includes extensive text preprocessing, emoji handling, normalization, and model training with transformer-based architectures.

## 🔍 Project Highlights

- Emoji and HTML character normalization
- Tokenization and lemmatization
- Spelling correction and slang expansion
- Named Entity Recognition (NER) filtering
- Sentiment classification using state-of-the-art models like BERT, RoBERTa
- Use of libraries like `transformers`, `tensorflow`, `gensim`, `emoji`, `symspellpy`, and more
- Handles multilingual and informal tweet text
- Model evaluation using metrics such as accuracy, precision, recall, and F1-score

## 🧰 Libraries and Tools Used

- **Transformers** (HuggingFace)
- **TensorFlow / Keras**
- **Pandas, NumPy**
- **NLTK, SpaCy, Gensim**
- **SymSpellPy**, **WordNinja**, **Contractions**
- **Tweepy** (for data extraction)
- **Emoji** and **pyspellchecker**
- **Joblib** for model saving
- **Datasets** from HuggingFace

## ⚙️ Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

> You may need to reinstall some core packages like `numpy`, `pandas`, and `scikit-learn` as done in the notebook to resolve dependency issues.

## 📊 Dataset

The dataset consists of tweets labeled for sentiment (positive/negative/neutral). Preprocessing steps clean and prepare the raw text data for modeling.

## 🧹 Preprocessing Pipeline

1. HTML decoding
2. Emoji conversion to text
3. Slang word replacement
4. Spelling correction
5. Lemmatization
6. Stopword removal
7. Named Entity removal
8. Lowercasing and punctuation handling

## 🧠 Model Training

Models are fine-tuned using HuggingFace Transformers. Includes:
- BERT
- RoBERTa
- DistilBERT (optional)

Each model is trained with cross-validation and hyperparameter tuning.

## 📈 Evaluation

The trained model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 💾 Model Persistence

Models are saved using `joblib` or HuggingFace's save utilities for reuse.

## 📎 Future Enhancements

- Real-time Twitter sentiment dashboard using Tweepy + Streamlit
- Multilingual support with XLM-RoBERTa
- Incorporation of sarcasm detection models

## 📁 File Structure

```
Twitter_data_sentiment_analysis.ipynb    # Main notebook
requirements.txt                         # List of packages (to be generated)
README.md                                # Project documentation
```
