import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from textblob import TextBlob
from nlp import NLP


# Create space between two contents:
def space():
    st.markdown("<br>", unsafe_allow_html=True)


# Heading
st.markdown("<h1 style='text-align: center; "
            "color: #3f3f44' >NLP Automation </h1>", unsafe_allow_html=True)
space()

# Sub-Heading
st.markdown("<strong><p style='color: #424874'>1) This project uses Naive Bayes Algorithm</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>2) You can choose different cleaning process (Stemming, "
            "Lemmatizing)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>3) Different type of  Metrics formation (Count Vectorizing, "
            "TF-IDF)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>4) Plotting Sentimental Analysis, Confusion Metrics and Word "
            "Cloud</p></strong>", unsafe_allow_html=True)
space()


# data preprocessing
def preprocess():
    try:
        preprocessing_option = ['Stemming', 'Lemmatization']
        preprocessor = st.radio("Select Preprocessing Technique", preprocessing_option,
                                horizontal=True)
        space()
        return preprocessor
    except Exception as e:
        print("preprocess ERROR : ", e)


# Hyperparameter Tuning
def hyperparameter_tuning():
    try:
        features = ["2500", "3000", "3500", "4000"]
        max_features = st.select_slider("Maximum features you want to restrict", features)
        space()
        ranges = ["1,1", "1,2", "1,3"]
        ngrams_range = st.radio("Combination of words", ranges, horizontal=True).split(",")
        return ngrams_range, max_features

    except Exception as e:
        print("hyperparameter ERROR : ", e)


# count vectorizer (Bag of Words) / TF-IDF
def bow():
    try:
        metrics = ['Count Vectorizer', 'TFIDF']
        bag_of_words = st.radio("Bag of Words", metrics, horizontal=False)
        return bag_of_words
    except Exception as e:
        print("boW ERROR : ", e)


# Converting Target Column
def y_label():
    try:
        target_column = ["Yes", "No"]
        y_option = st.selectbox("Do you want to encode your target column", target_column)
        return y_option
    except Exception as e:
        print("y_label ERROR : ", e)


# Main Function
def app():
    df = st.file_uploader("Upload your dataset", type=['CSV', "txt"])
    space()

    if df is not None:
        data = pd.read_csv(df, encoding="ISO-8859-1")
        st.dataframe(data.sample(5))
        space()

        text = st.selectbox("Select text Column", data.columns)
        space()

        target = st.selectbox("Select Target Column", data.columns)
        space()

        # Reassigning features to Dataframe
        data = data[[text, target]]

        # Drop Nan Values
        data = data.dropna()

        nlp_model = NLP(data)

        st.markdown("<h4 style='color: #438a5e'>Final Dataset</h4>", unsafe_allow_html=True)
        st.dataframe(data.head())
        space()

        # Calling functions for preprocessing, Bag of Words, target variable

        ngram_range, max_features = hyperparameter_tuning()
        space()
        space()
        bag_of_words = bow()
        space()
        space()
        y_option = y_label()
        space()
        space()

        # define function
        def matrix(corpus, bag_of_words, ngram_range, max_features):
            try:
                if bag_of_words == 'Count_Vectorizer':
                    X = nlp_model.Count_Vectorizer(corpus,
                                                   int(max_features),
                                                   (int(ngram_range[0]),
                                                    int(ngram_range[1])))
                    return X
                elif bag_of_words == 'TF_IDF':
                    X = nlp_model.TF_IDF(corpus,
                                         int(max_features),
                                         (int(ngram_range[0]),
                                          int(ngram_range[1])))
                    return X
            except Exception as e:
                print("metrix Error : ", e)

        def target_series(y_option, target):
            try:
                if y_option == "Yes":
                    y = nlp_model.y_encoding(target)
                    return y

                elif y_option == "No":
                    y = data[target]
                    return y
            except Exception as e:
                print("target_series ERROR : ", e)

        def plot_word_cloud(corpus, y_test, y_pred):
            st.success('Word Cloud')
            word_cloud = nlp_model.word_cloud(corpus)
            st.image(word_cloud)
            accuracy, confusion_matrix = nlp_model.confusion_matrix(y_test, y_pred)
            st.success(f"Accuracy: {round(accuracy * 100, 2)}%")
            st.image(confusion_matrix)

        def sentiment_analysis(text):
            data["sentiments"] = data[text].apply(nlp_model.sentimental_analysis_clean)

            def get_subjectivity(text):
                return TextBlob(text).sentiment.subjectivity

            def get_polarity(text):
                return TextBlob(text).sentiment.polarity

            def get_analysis(score):
                if score < 0:
                    return "Negative"
                elif score == 0:
                    return "Neutral"
                else:
                    return "Positive"

            data["Subjectivity"] = data['sentiments'].apply(get_subjectivity)
            data["Polarity"] = data['sentiments'].apply(get_polarity)
            data["Analysis"] = data['Polarity'].apply(get_analysis)

            st.success("Sentiments")
            sns.countplot(x=data['Analysis'])
            st.pyplot(use_container_width=True)

        # Model Creation
        if st.button("Submit"):
            space()
            if preprocess == 'Stemming':
                corpus = nlp_model.stemming(text)
                X = matrix(corpus, bag_of_words, max_features, ngram_range)
                y = target_series(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_bayes(X_train, X_test, y_train, y_test)
                sentiment_analysis(text)
                plot_word_cloud(corpus, y_test, y_pred)

            else:
                corpus = nlp_model.Lemmatization(text)
                X = matrix(corpus, bag_of_words, max_features, ngram_range)
                y = target_series(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_bayes(X_train, X_test, y_train, y_test)
                sentiment_analysis(text)
                plot_word_cloud(corpus, y_test, y_pred)


if __name__ == "__main__":
    app()
