import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")

st.title("FAQ Chatbot")
st.write("Welcome! Ask a question about AI and programming.")

faqs = {
    "What is artificial intelligence?":
        "Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.",

    "What is machine learning?":
        "Machine Learning is a branch of AI that helps computers learn patterns from data.",

    "What is Python?":
        "Python is a beginner-friendly programming language used in AI, web development, and data science.",

    "What is NLP?":
        "Natural Language Processing (NLP) helps computers understand human language.",

    "What is deep learning?":
        "Deep Learning uses neural networks with multiple layers to learn complex patterns.",

    "What is data science?":
        "Data Science uses data analysis, statistics, and programming to discover useful insights."
}

questions = list(faqs.keys())
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip():
        user_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(user_vector, question_vectors)
        best_match = scores.argmax()

        if scores[0][best_match] >= 0.2:
            st.success(faqs[questions[best_match]])
        else:
            st.warning("Sorry, I don't know that answer yet. Try another question.")
    else:
        st.warning("Please enter a question.")

st.subheader("Frequently Asked Questions")
for question in questions:
    with st.expander(question):
        st.write(faqs[question])