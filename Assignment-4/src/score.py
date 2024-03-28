#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:56:33 2024

@author: aryan
"""


import pickle
import string
from nltk.corpus import stopwords
import nltk



from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('./vectorizer.pkl','rb'))



def score(text, model, threshold):
    # 1. preprocess
    transformed_sms = transform_text(text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict_proba(vector_input)[0]
    propensity = result[1]
    prediction = propensity > threshold
    
    return prediction.item(), propensity