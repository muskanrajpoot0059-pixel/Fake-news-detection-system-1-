import requests
import streamlit as st
import pickle
import pandas as pd
import re
import string

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]




# -----------------------------
# Load Models
# -----------------------------
LR = pickle.load(open("LR.pkl", "rb"))
DT = pickle.load(open("DT.pkl", "rb"))
RFC = pickle.load(open("RFC.pkl", "rb"))
vectorization = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Text Cleaning Function
# Replace this with the exact word_drop()
# function from your Colab notebook if needed.
# -----------------------------
def word_drop(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -----------------------------
# Prediction Function
# -----------------------------

def predict_news(news):

    news = word_drop(news)

    news = vectorization.transform([news])

    lr_pred = LR.predict(news)[0]
    dt_pred = DT.predict(news)[0]
    rfc_pred = RFC.predict(news)[0]

    predictions = [lr_pred, dt_pred, rfc_pred]

    final = max(set(predictions), key=predictions.count)

    return final, lr_pred, dt_pred, rfc_pred



# -----------------------------
# Online Verification
# -----------------------------
def verify_online(news):
     

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": news[:60],
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])

    return []

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)
st.markdown("""
<style>

/* Main page */
.main {
    background-color: #f5f7fa;
}

/* Center the content */
.block-container {
    max-width: 850px;
    margin: auto;
    padding-top: 2rem;
}

/* Title */
h1 {
    text-align: center;
    color: #1E3A8A;
}

/* Subheadings */
h2, h3 {
    color: #2563EB;
}

/* Text Area */
textarea {
    border-radius: 12px !important;
}

/* Button */
.stButton>button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    font-size: 15px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>📰 Fake News Detection System</h1>
<h4 style='text-align:center;color:gray;'>
Machine Learning Based News Classification
</h4>
""", unsafe_allow_html=True)

st.write("### Enter a news article below")

news = st.text_area(
    "Paste News Here",
    height=250
)


if st.button("Predict"):

    if news.strip() == "":
        st.warning("⚠ Please enter a news article.")

    else:

        # ML Prediction
        final, lr, dt, rfc = predict_news(news)

        st.markdown("---")
        st.subheader("📊 Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Logistic Regression",
                "Real" if lr == 1 else "Fake"
            )

        with col2:
            st.metric(
                "Decision Tree",
                "Real" if dt == 1 else "Fake"
            )

        with col3:
            st.metric(
                "Random Forest",
                "Real" if rfc == 1 else "Fake"
            )

        st.markdown("---")
        st.subheader("🎯 Final Prediction")

        if final == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
         # -----------------------------
        # Online Verification
        # -----------------------------
        st.markdown("---")
        st.subheader("🌐 Online Verification")

        articles = verify_online(news)

        if len(articles) == 0:
            st.warning("No similar news found online.")
        else:
            st.success(f"Found {len(articles)} similar articles")

            for article in articles:
                st.write("### 📰", article["title"])
                st.write("**Source:**", article["source"]["name"])
                st.write("**Published:**", article["publishedAt"])
                st.markdown(f"[🔗 Read Full Article]({article['url']})")
                st.markdown("---")
 
   
    
            # -----------------------------
       


st.markdown("""
<div class='footer'>
Developed by <b>Muskan Rajpoot</b> ❤️
</div>
""", unsafe_allow_html=True)
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2965/2965879.png",
        width=100
    )

    st.title("Fake News Detection")

    st.markdown("---")

    st.write("### Models Used")

    st.write("✔ Logistic Regression")
    st.write("✔ Decision Tree")
    st.write("✔ Random Forest")

    st.markdown("---")

    st.write("### Dataset")

    st.write("• Fake.csv")
    st.write("• True.csv")

    st.markdown("---")

    st.write("### Developer")

    st.success("Muskan Rajpoot")