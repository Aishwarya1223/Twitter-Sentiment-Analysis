
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

> You may need to reinstall some core packages like `numpy`, `pandas`, and `scikit-learn` as done in the notebook to resolve dependency issues.

## 📊 Dataset

The dataset, sourced from Kaggle, contains tweets labeled with sentiment categories: positive, negative, or neutral. The raw text data undergoes preprocessing to clean and prepare it for model training.

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
This project includes the development and fine-tuning of both custom and pre-trained models for text analysis:

🔧 Custom-Built Models:
- LSTM – A standard Long Short-Term Memory network for sequence modeling.

- BiLSTM – A bidirectional LSTM to capture context from both directions.

- BiLSTM + Self-Attention – Enhanced with a self-attention mechanism to focus on the most relevant parts of the input sequence.

🤗 Fine-Tuned Transformer Models (via Hugging Face):
- BERT – Bidirectional Encoder Representations from Transformers.

- RoBERTa – A robustly optimized version of BERT.

- DistilBERT – A lightweight and faster version of BERT with minimal performance trade-off.

These models were trained and evaluated for high performance on the target NLP tasks.

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
