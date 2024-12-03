import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Model va vectorizerni yuklash
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Tfidf vectorizerni yuklash
vectorizer = TfidfVectorizer(max_features=5000)

# Web ilovasi interfeysi
st.title('Sentiment Analysis Web App')

st.write("""
    Bu ilova film sharhining ijobiy yoki salbiy ekanligini aniqlaydi.
    Iltimos, quyidagi matn maydoniga sharhni kiriting va "Predict" tugmasini bosing.
""")

# Foydalanuvchidan input olish
user_input = st.text_area("Enter a movie review:")

if st.button('Predict'):
    if user_input:
        # Matnni vektorlashtirish
        input_tfidf = vectorizer.transform([user_input])
        
        # Natijani bashorat qilish
        prediction = model.predict(input_tfidf)
        
        # Natijani chiqarish
        if prediction[0] == 1:
            st.success('Positive Sentiment')
        else:
            st.error('Negative Sentiment')
    else:
        st.warning("Iltimos, sharhni kiriting.")
