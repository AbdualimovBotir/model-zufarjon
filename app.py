import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Model va vectorizerni yuklash (birgalikda saqlanadi)
with open('sentiment_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)  # Model va vectorizer birgalikda saqlangan

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
        # Matnni vektorlashtirish (fit bo'lgan vectorizer bilan)
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
