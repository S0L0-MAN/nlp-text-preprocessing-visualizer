import streamlit as st
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from collections import Counter
import pandas as pd

nltk.download('punkt_tab')
st.set_page_config(page_title="NLP Preprocessing Visualizer",layout='wide')
st.title("NLP Preprocessing Visualization")

text_input = st.text_area("Enter your text here:","My grandfather has the heart of a lion and a lifetime ban from the zoo")

if st.button("Visualize"):
    st.subheader("PreProcessing Steps ->")
    

    time.sleep(1)
    lowered_text = text_input.lower()
    st.empty().markdown(f"**First Step is to lowercase the acquired text** `{lowered_text}`")

    time.sleep(1.5)
    tokenized_text = word_tokenize(lowered_text)
    st.empty().markdown(f"**Second Step is to tokenized the lowered text** `{tokenized_text}`")

    time.sleep(2)
    stop_words = set(stopwords.words('english'))
    no_stop = [w for w in tokenized_text if w not in stop_words]
    st.empty().markdown(f"**Third Step is to remove stopwords the tokenized text** `{no_stop}`")

    time.sleep(2.5)
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in no_stop]
    st.empty().markdown(f"**Fourth Step is to perform stemming on the text:** `{stemmed_words}`")

    time.sleep(3)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in no_stop]
    st.empty().markdown(f"**Fifth Step is to perform lemmatization on the text:** `{lemmatized_words}`")

    time.sleep(3.5)
    st.subheader("Word Frequencies")

    col1, col2 = st.columns(2)

    time.sleep(1)
    with col1:
        st.markdown("**Before Preprocessing**")
        freq_all = Counter(tokenized_text)
        df_all = pd.DataFrame(freq_all.items(), columns=["Word", "Count"]).set_index("Word")
        st.bar_chart(df_all)
    
    
    time.sleep(1.5)
    with col2:
        st.markdown("**After Preprocessing**")
        freq_clean = Counter(lemmatized_words)
        df_clean = pd.DataFrame(freq_clean.items(), columns=["Word", "Count"]).set_index("Word")
        st.bar_chart(df_clean)




    



