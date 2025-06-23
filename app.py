import streamlit as st
import pickle
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Load model dan TF-IDF vectorizer
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load kamus normalisasi
import pandas as pd
df_kamus = pd.read_excel('kamusbakutidakbakufix.xlsx')
kamus_normalisasi = dict(zip(df_kamus['tidakbaku'], df_kamus['baku']))

# Preprocessing
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub(r'\d+', '', tweet)
    return tweet.strip()

def preprocess(tweet):
    tweet = clean_tweet(tweet)
    tweet = tweet.lower()
    tokens = tweet.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [kamus_normalisasi.get(token, token) for token in tokens]
    return ' '.join(tokens)

def predict_sentiment(tweet):
    processed = preprocess(tweet)
    st.write("Tweet setelah preprocessing:", processed)
    tfidf_vector = tfidf.transform([processed]).toarray()  # ubah jadi dense
    pred = model.predict(tfidf_vector)[0]
    return pred

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Tweet")

tweet_input = st.text_area("Masukkan Tweet")

if st.button("Analisis Sentimen"):
    if tweet_input:
        label = predict_sentiment(tweet_input)
        st.success(f"Sentimen untuk tweet adalah **{label}**")
    else:
        st.warning("Silakan masukkan tweet terlebih dahulu.")
