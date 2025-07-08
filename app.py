import streamlit as st
import pandas as pd
import nltk
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import nlp_utils as nu

# Setup
nltk.download('vader_lexicon')
nltk.download('punkt')
st.set_page_config(page_title="ChatInsight", layout="centered")

# Load and prepare data
df = pd.read_csv('dialogs.txt', names=('Query', 'Response'), sep='\t')
sid = SentimentIntensityAnalyzer()

df['pos'] = df['Query'].apply(lambda x: sid.polarity_scores(x)['pos'])
df['neg'] = df['Query'].apply(lambda x: sid.polarity_scores(x)['neg'])
df['neu'] = df['Query'].apply(lambda x: sid.polarity_scores(x)['neu'])

tfidf = TfidfVectorizer()
factors = tfidf.fit_transform(df['Query'])

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar sentiment summary
st.sidebar.title("📊 Sentiment Tracker")
if st.session_state.chat_history:
    last_user_input = [entry for entry in st.session_state.chat_history if entry['sender'] == 'You'][-1]['text']
    sentiment = sid.polarity_scores(last_user_input)
    st.sidebar.markdown(f"**Last Query Sentiment:**")
    st.sidebar.write(f"😊 Positive: {sentiment['pos']}")
    st.sidebar.write(f"😐 Neutral: {sentiment['neu']}")
    st.sidebar.write(f"😠 Negative: {sentiment['neg']}")
else:
    st.sidebar.info("Chat to see sentiment here!")

# Chatbot function
def chatbot(query):
    cleaned = nu.lemmatization_sentence(query)
    query_vec = tfidf.transform([cleaned]).toarray()
    score = 1 - cosine_distances(factors, query_vec)
    index = score.argmax()
    confidence = score[index][0]
    if confidence <= 0.2:
        return "Please rephrase your question.", confidence
    return df.loc[index, 'Response'], confidence

# Header and input
st.title("💬 ChatInsight")
st.write("Talk to your intelligent bot. Type a query below:")

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("You:", key="input", label_visibility="collapsed")
with col2:
    clear = st.button("🗑️ Clear")

# Clear chat history
if clear:
    st.session_state.chat_history = []

# Process input
if user_input:
    response, conf = chatbot(user_input)
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({'sender': 'You', 'text': user_input, 'time': now})
    st.session_state.chat_history.append({'sender': 'Bot', 'text': response, 'time': now})

# Display chat history
for entry in st.session_state.chat_history:
    if entry['sender'] == 'You':
        st.markdown(f"🧑 **You** ({entry['time']}): {entry['text']}")
    else:
        st.markdown(f"🤖 **Bot** ({entry['time']}): {entry['text']}")
