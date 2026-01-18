import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DeadMLSociety", page_icon="🎬", layout="centered")
 
st.markdown("""
    <style>
    /* Background Color */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Title Style WITH ANIMATION */
    h1 {
        font-family: 'Playfair Display', serif;
        color: #e50914 !important;
        text-shadow: 0 0 20px rgba(229, 9, 20, 0.6);
        text-align: center;
        font-size: 4rem !important;
        animation: logoIntro 2s ease-out forwards;
    }
    
    /* Animation Keyframes */
    @keyframes logoIntro {
        0% {
            opacity: 0;
            transform: scale(0.5);
            letter-spacing: 20px;
        }
        80% {
            opacity: 1;
            transform: scale(1.1);
            letter-spacing: 2px;
        }
        100% {
            transform: scale(1);
            letter-spacing: normal;
        }
    }

    /* Subtitle/Text */
    p, label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
    }
    /* Input Box */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #e50914;
        border-radius: 10px;
    }
    /* Button Style */
    .stButton > button {
        background-color: transparent;
        color: #e50914;
        border: 1px solid #e50914;
        border-radius: 12px;
        padding: 10px 24px;
        font-family: 'Playfair Display', serif;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: rgba(229, 9, 20, 0.1);
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.6);
        color: white;
    }
    /* Movie Cards */
    .movie-card {
        background-color: rgba(229, 9, 20, 0.05);
        border: 1px solid rgba(229, 9, 20, 0.3);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(229, 9, 20, 0.8);
    }
    h3 {
        color: #e50914 !important;
        margin-bottom: 5px;
    }
    /* Custom Footer Button */
    .custom-link-btn {
        display: inline-block;
        text-decoration: none;
        background-color: transparent;
        color: #e50914;
        border: 1px solid #e50914;
        border-radius: 12px;
        padding: 12px 28px;
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-align: center;
    }
    .custom-link-btn:hover {
        background-color: rgba(229, 9, 20, 0.1);
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.6);
        color: white;
        border-color: #ff4444;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# LOAD MODELS 
@st.cache_resource
def load_data():
    try:
        
        vec = joblib.load('vectorizer.pkl')
        mat = joblib.load('matrix.pkl')
        mov = joblib.load('movies.pkl') 
        return vec, mat, mov
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

vectorizer, matrix, movies = load_data()

# THE UI
st.title("DeadMLSociety")
st.markdown("<p style='text-align: center; font-style: italic; margin-bottom: 40px;'> Show me a query unfettered by hidden desire, and I'll show you a soulless search.</p>", unsafe_allow_html=True)

if movies is not None:
    user_input = st.text_area("Decode your conversation:", height=100, placeholder="Type here (e.g., I want a sad romantic movie)...")

    if st.button("ANALYZE"):
        if user_input.strip():
            with st.spinner("Decoding the films within your words..."):
                try:
                    
                    user_vec = vectorizer.transform([user_input])
                    similarity = cosine_similarity(user_vec, matrix).flatten()
                    indices = similarity.argsort()[-5:][::-1]
                    results = movies.iloc[indices]
                    
                    st.markdown("---")
                    st.subheader("Recommended Films")
                    
                    for _, row in results.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>{row['title']}</h3>
                            <p style="color: grey; font-size: 0.9rem;">{row['genres']}</p>
                            <p style="font-style: italic;">A match based on your conversation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text first.")
else:
    st.error("⚠️ Files missing! Please upload vectorizer.pkl, matrix.pkl, and movies.pkl to GitHub.")
st.markdown("---") 
st.subheader("Explore More")
st.link_button("Visit DeadMLSociety ↗", "https://deadml-society.streamlit.app")