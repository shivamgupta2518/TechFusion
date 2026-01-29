import streamlit as st
import joblib
import pandas as pd
import gmail

spam_model = joblib.load("spam_classifier.pkl")
language_model = joblib.load("lang_det.pkl")
news_model = joblib.load("news_cat.pkl")
review_model = joblib.load("review.pkl")

st.set_page_config(layout="wide")


st.markdown(
    """
    <h1 style="
        background-color: #DA70D6;  /* background color */
        color: white;                /* text color */
        padding: 15px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
        TextFusion
    </h1>
    """,
    unsafe_allow_html=True,
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🤖 Spam Classifier",
        "🗣️ Language Detection",
        "🍴 Food Review Sentiment",
        "📰 News Classification",
    ]
)
with tab1:
    msg = st.text_input("Enter Msg")
    if st.button("Prediction"):
        pred = spam_model.predict([msg])
        if pred[0] == 0:
            st.image("spam_image.jpg")
        else:
            st.image("not_spam.jpg")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

    if uploaded_file:

        df_spam = pd.read_csv(uploaded_file, header=None, names=["Msg"])

        pred = spam_model.predict(df_spam.Msg)
        df_spam.index = range(1, df_spam.shape[0] + 1)

        df_spam["Prediction"] = pred
        df_spam["Prediction"] = df_spam["Prediction"].map({0: "spam", 1: "Not Spam"})
        st.dataframe(df_spam)


with tab2:
    text = st.text_area("Enter language")

    if st.button("Detect Language"):
        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            pred = language_model.predict([text])[0]
            lang_map = {
                "en": "English",
                "hi": "Hindi",
                "fr": "French",
                "de": "German",
                "es": "Spanish",
                "ur": "Urdu",
                "ta": "Tamil",
                "te": "Telugu",
            }

            detected_language = lang_map.get(pred, pred)
            st.success(f"Detected Language: **{detected_language}**")

    uploaded_file = st.file_uploader(
        "Upload file for language detection", type=["csv", "txt"], key="lang_upload"
    )

    if uploaded_file:
        df_lang = pd.read_csv(uploaded_file, header=None, names=["Text"])

        preds = language_model.predict(df_lang["Text"])

        df_lang.index = range(1, len(df_lang) + 1)
        df_lang["Language"] = preds

        df_lang["Language"] = (
            df_lang["Language"].map(lang_map).fillna(df_lang["Language"])
        )

        st.dataframe(df_lang)


with tab3:
    review = st.text_input("Enter Restaurant Review")

    if st.button("Prediction", key="review_btn"):
        pred = review_model.predict([review])

        if pred[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "txt"], key="review_upload"
    )

    if uploaded_file:
        df_review = pd.read_csv(uploaded_file, header=None, names=["Review"])

        pred = review_model.predict(df_review.Review)
        df_review.index = range(1, df_review.shape[0] + 1)

        df_review["Prediction"] = pred
        df_review["Prediction"] = df_review["Prediction"].map(
            {0: "Negative", 1: "Positive"}
        )

        st.dataframe(df_review)


with tab4:
    headline = st.text_input("Enter News Headline")

    if st.button("Prediction", key="news_btn"):
        if not headline.strip():
            st.warning("Please enter a news headline")
        else:
            pred = news_model.predict([headline])

            news_map = {
                0: "Politics",
                1: "Sports",
                2: "Business",
                3: "Technology",
                4: "Entertainment",
                5: "Comedy",
                6: "Accident",
                7: "Crime",
                8: "Health",
                9: "Education",
            }

            category = news_map.get(pred[0], pred[0])
            st.success(f"📰 News Category: **{category}**")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "txt"], key="news_upload"
    )

    if uploaded_file:
        df_news = pd.read_csv(uploaded_file, header=None, names=["Headline"])

        pred = news_model.predict(df_news["Headline"])
        df_news.index = range(1, df_news.shape[0] + 1)

        df_news["Prediction"] = pred
        df_news["Prediction"] = df_news["Prediction"].map(news_map).fillna("Unknown")

        st.dataframe(df_news)


st.sidebar.image("S:/machine learning/Machine Learning aditya sir/NLP/flage.svg")


with st.sidebar.expander("🧑‍💼 About us"):
    st.write(
        "TextFusion is an all-in-one NLP platform that analyzes text using four AI models. It can detect spam messages, identify languages, predict review sentiment, and classify news headlines into categories. The platform provides fast, accurate, and actionable insights for students, researchers, and businesses."
    )

with st.sidebar.expander("📞 Contact us"):
    st.write("9999999999")
    st.write("aa@gmail.com")
