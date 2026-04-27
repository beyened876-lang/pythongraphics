import os
import pickle
import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    'lr_model': 'lr_model.pkl',
    'rf_model': 'rf_model.pkl',
    'lstm_model': 'lstm_model.h5',
    'tfidf': 'tfidf.pkl',
    'tokenizer': 'tokenizer.pkl',
    
}


def data_path(filename):
    return os.path.join(BASE_DIR, filename)


def safe_download_nltk():
    try:
        stopwords.words('english')
        WordNetLemmatizer()
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)


def load_models():
    missing = [name for name, file in MODEL_FILES.items() if not os.path.exists(data_path(file))]
    if missing:
        raise FileNotFoundError(
            'Missing saved model files: ' + ', '.join(MODEL_FILES[name] for name in missing)
        )

    with open(data_path(MODEL_FILES['lr_model']), 'rb') as f:
        lr_model = pickle.load(f)
    with open(data_path(MODEL_FILES['rf_model']), 'rb') as f:
        rf_model = pickle.load(f)
    with open(data_path(MODEL_FILES['tfidf']), 'rb') as f:
        tfidf = pickle.load(f)
    with open(data_path(MODEL_FILES['tokenizer']), 'rb') as f:
        tokenizer = pickle.load(f)

    lstm_model = load_model(data_path(MODEL_FILES['lstm_model']))
    return lr_model, rf_model, tfidf, tokenizer, lstm_model


@st.cache_resource
def get_models():
    safe_download_nltk()
    return load_models()


def preprocess_text(text):
    if not isinstance(text, str):
        return ''

    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    )


def load_test_data():
    test_df = pd.read_csv('test.tsv', sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    return test_df


@st.cache_data
def predict_news_cached(text, model_type, _tfidf, _tokenizer, _lr_model, _rf_model, _lstm_model):
    clean_text = preprocess_text(text)
    if model_type == 'Logistic Regression':
        vector = _tfidf.transform([clean_text])
        prob = float(_lr_model.predict_proba(vector)[0][1])
    elif model_type == 'Random Forest':
        vector = _tfidf.transform([clean_text])
        prob = float(_rf_model.predict_proba(vector)[0][1])
    else:
        seq = _tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=100)
        prob = float(_lstm_model.predict(padded, verbose=0)[0][0])

    label = 'Real' if prob >= 0.5 else 'Fake'
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return label, confidence


def main():
    st.set_page_config(page_title='Fake News Detection System', layout='centered')
    
    # Custom CSS for colorful interface
    st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stTitle {
        color: #ff4500;
        font-family: 'Arial Black', sans-serif;
    }
    .stTextArea, .stSelectbox, .stButton {
        background-color: #fffacd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title('📰 Fake News Detection System')
    st.write('Paste a news statement below for detection, or classify all statements from the folder dataset.')

    try:
        lr_model, rf_model, tfidf, tokenizer, lstm_model = get_models()
        test_df = load_test_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info('Run `python model_training.py` first to create the saved model files.')
        return
    except Exception as e:
        st.error(f'Error loading data: {e}')
        return

    user_statement = st.text_area('Paste news statement here:', height=180)
    model_choice = st.selectbox('Choose Model', ['Logistic Regression', 'Random Forest', 'LSTM'])

    if st.button('Detect Statement'):
        if not user_statement.strip():
            st.warning('Please paste a statement before detecting.')
        else:
            with st.spinner('Analyzing pasted statement...'):
                result, confidence = predict_news_cached(
                    user_statement, model_choice, tfidf, tokenizer, lr_model, rf_model, lstm_model
                )
            if result == 'Real':
                st.success(f'✅ REAL NEWS: {confidence * 100:.1f}% confidence')
            else:
                st.error(f'❌ FAKE NEWS: {confidence * 100:.1f}% confidence')

    if st.button('Classify All Test Statements'):
        with st.spinner('Classifying all statements...'):
            results = []
            for idx, row in test_df.iterrows():
                statement = row['statement']
                true_label = row['label']
                result, confidence = predict_news_cached(
                    statement, model_choice, tfidf, tokenizer, lr_model, rf_model, lstm_model
                )
                results.append({
                    'ID': row['id'],
                    'Statement': statement[:100] + '...' if len(statement) > 100 else statement,
                    'True Label': true_label,
                    'Predicted': result,
                    'Confidence': f'{confidence * 100:.1f}%',
                    'Correct': 'Yes' if (result == 'Real' and true_label not in ['false', 'pants-fire']) or (result == 'Fake' and true_label in ['false', 'pants-fire']) else 'No'
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Summary
            total = len(results)
            correct = sum(1 for r in results if r['Correct'] == 'Yes')
            accuracy = correct / total * 100
            st.success(f'Overall Accuracy: {accuracy:.2f}% ({correct}/{total})')


if __name__ == '__main__':
    main()