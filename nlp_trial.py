import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import streamlit as st
st.title("NLP Project")
st.write("Enter the text which need s to be cleaned in the text box:")
text=st.text_input(label="Enter text")

if text:
    tokens=word_tokenize(text=text)
    stemmer=PorterStemmer()
    lemme=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    stem_words=[]
    lemmetize_words=[]
    for word in tokens :
        if word.lower() not in stop_words:
            stem_word=stemmer.stem(word) 
            lemme_word=lemme.lemmatize(word)
            stem_words.append(stem_word)
            lemmetize_words.append(lemme_word)
    st.write("stemmed words",stem_words)
    st.write("lemmeatized words",lemmetize_words)
    pos_tag=nltk.pos_tag(lemmetize_words)
    st.write("Parts of speech tagging",pos_tag)