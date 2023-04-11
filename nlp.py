import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import scikitplot as skplt

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from exception import CustomException
from logger import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from wordcloud import WordCloud


class NLP:
    def __init__(self, data):
        self.data = data

    def stemming(self, column_name):
        try:
            logging.info("Stemming process is Initiated")
            corpus = []
            stemming = PorterStemmer()
            for i in range(len(self.data)):
                tweet = re.sub("[^a-zA-Z]", " ", self.data[column_name][i])
                tweet = re.sub("http", '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [stemming.stem(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet = "".join(tweet)
                corpus.append(tweet)
            logging.info('Stemming is finished.')
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return corpus
        finally:
            logging.info('Stemming is finished.')

    def Lemmatization(self, column_name):
        try:
            logging.info("Lemmatization process is Initiated")
            corpus = []
            lemmatizing = WordNetLemmatizer()
            for i in range(len(self.data)):
                tweet = re.sub("[^a-zA-Z]", " ", self.data[column_name][i])
                tweet = re.sub("http", '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatizing.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet = "".join(tweet)
                corpus.append(tweet)
            logging.info('Lemmatization is finished.')
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return corpus
        finally:
            logging.info('Lemmatization  is finished.')

    @staticmethod
    def Count_Vectorizer(self, corpus, max_features=3000, ngram_range=(1, 2)):
        # Bag of Words
        try:
            logging.info("Count Vectorizer process is started")
            cv = CountVectorizer(max_features=max_features,
                                 ngram_range=ngram_range)
            X = cv.fit_transform(corpus).toarray()
            logging.info("Count Vectorizer process is Successful.")
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return X
        finally:
            logging.info('Count Vectorizer is finished.')

    @staticmethod
    def TF_IDF(self, corpus, max_features=3000, ngram_range=(1, 2)):
        # Bag of Words
        try:
            logging.info("TF_IDF process is started")
            tf_idf = TfidfVectorizer(max_features=max_features,
                                     ngram_range=ngram_range)
            X = tf_idf.fit_transform(corpus).toarray()
            logging.info("TF_IDF process is Successful.")
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return X
        finally:
            logging.info('TF_IDF is finished.')

    @staticmethod
    def y_encoding(self, target_label):
        try:
            y = pd.get_dummies(self.data[target_label], drop_first=True)
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return y
        finally:
            logging.info('Target encoding is finished.')

    @staticmethod
    def split_data(self, X, y, test_size=0.25, random_state=0):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=test_size,
                                                                random_state=random_state)
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return X_train, X_test, y_train, y_test

    @staticmethod
    def naive_bayes(self, X_train, X_test, y_train, y_test):
        try:
            naive = MultinomialNB()
            naive.fit(X_train, y_train)
            y_pred = naive.predict(X_test)
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return y_pred

    @staticmethod
    def confusion_matrix(self, y_test, y_pred):
        try:
            skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                                figsize=(8, 7))
            plt.savefig("Confusion_Matrix.jpg")
            image_cm = Image.open("Confusion_Matrix.jpg")
            accuracy = accuracy_score(y_test, y_pred)
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return accuracy, image_cm

    @staticmethod
    def word_cloud(self, corpus):
        try:
            word_cloud = WordCloud(background_color='white',
                                   width=750,
                                   height=500).generate(" ".join(corpus))
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis('off')
            plt.savefig("Word_cloud.jpg")
            img = Image.open("Word_cloud.jpg")
        except Exception as e:
            raise CustomException(e, sys) from e
        else:
            return img
