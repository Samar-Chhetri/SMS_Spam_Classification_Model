import streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string


# Function for text-preprocessing
def transform_text(text):
    text = text.lower()   # Lower-case
    text = nltk.word_tokenize(text)   # Tokenize the text on  word level
    
    y = []
    for i in text:
        if i.isalnum():    # Check whether the text is either alpha or numeric
            y.append(i)    # If yes, append otherwise, remove special characters
            
    text = y[:]
    y.clear()
    
    for j in text:    # Removing stopwords
        if j not in stopwords.words('english') and j not in string.punctuation:
            y.append(j)
            
    text = y[:]
    y.clear()
    
    for k in  text:
        y.append(ps.stem(k))    # Convert the word to root form
    
    return " ".join(y)

tfidf = pickle.load(open('notebook/vectorizer.pkl', 'rb'))
model = pickle.load(open('notebook/model.pkl', 'rb'))




st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # Preprocess        
    transform_sms = transform_text(input_sms)

    # step 2: vectorize
    vector_input = tfidf.transform([transform_sms])

    # step 3: predict
    result = model.predict(vector_input)[0]

    # step 4: disply
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


