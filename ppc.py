import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas_profiling
import json
import glob
import json_lines
from tqdm.notebook import tqdm
import re
import string
from string import punctuation
import preprocessor as p
from nltk.stem import *
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
##define some functions
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION)
def umlaut(text):
    text = text.replace('ä', 'aeqwe')
    text = text.replace('ö', 'oeqwe')
    text = text.replace('ü', 'ueqwe')
    text = text.replace('ß', 'ssqwe')
    return text
def clean_tweet(text):
    text = p.clean(text)
    return text
def remove_rt(text):
    if text.startswith('rt'):
        text = text[4:]  
    return text
def remove_punkt(text):
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text.replace("#", "").replace("_", " ")
    return text

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else '' for i in text])

def re_umlaut(text):
    text = text.replace('aeqwe', 'ä')
    text = text.replace('oeqwe', 'ö')
    text = text.replace('ueqwe', 'ü') 
    text = text.replace('ssqwe', 'ß')
    return text

st.title('Political Party Classification')
#present your project
st.subheader("Our Goal")
st.text('The goal of our project is to classify Tweets by the political party of the author')

#present your team
with st.expander("Our Team:"):
    col1, col2, col3 = st.columns(3)
with col1:
    st.image("image_jan.jpeg", caption = 'Jan Amend')
with col2:
    st.image("image.jpeg", caption = "Jana Adler")

with col3:
    st.image("image_ser.jpg", caption = 'Sergei Mezhonnov')

##Extract text of tweet and name of party 
#     tweets = []
#     party = ""
#     text = ""
#     files = glob.glob('data/*.jl')
#     for file in files:
#         with json_lines.open(file) as jl:
#             for item in jl:
#                 party = item['account_data']['Partei']
#                 if 'response' in item:
#                     if 'data' in item['response']:
#                         for i in item['response']['data']:
#                             tweet = re_umlaut(remove_punkt((remove_rt(clean_tweet(umlaut(i['text'].lower()))))))
#                             tweet = tweet.strip()
#                             if (len(tweet) > 5):
#                                 tweets.append({'party': party, 'tweet': tweet})

#     df = pd.DataFrame(tweets, columns = ['party', 'tweet'])
#     df.to_csv('tweets.csv', columns = ['party', 'tweet'], index=False)

#the same number of tweets per party
# party = []
# count = [0,0,0,0,0,0,0,0]
# i = 0
# #0 - Bündnis 90/Die Grünen, 1 - SPD, 2 - AfD, 3 - Die Linke, 4 - FDP, 5 - CSU, 6 - CDU, 7 - Fraktionslos
# for index,row in df.iterrows():
#     if row[0] == "Bündnis 90/Die Grünen":
#         if count[0] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[0] = count[0] + 1
#             i +=1
#     elif row[0] == "SPD":
#         if count[1] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[1] = count[1] + 1
#             i +=1     
#     elif row[0] == "AfD":
#         if count[2] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[2] = count[2] + 1
#             i +=1
#     elif row[0] == "Die Linke":
#         if count[3] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[3] = count[3] + 1
#             i +=1
#     elif row[0] == "FDP":
#         if count[4] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[4] = count[4] + 1
#             i +=1
#     elif row[0] == "CSU":
#         if count[5] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[5] = count[5] + 1
#             i +=1
#     elif row[0] == "CDU":
#         if count[6] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[6] = count[6] + 1
#             i +=1
#     elif row[0] == "Fraktionslos":
#         if count[7] < 2000:
#             party.append({'party': row[0], 'tweet': row[1]})
#             count[7] = count[7] + 1
#             i +=1  
# df1 = pd.DataFrame(party, columns = ['party', 'tweet'])   
# df1.to_csv('result.csv', columns = ['party', 'tweet'], index=False)
##datei einlesen
df = pd.read_csv('result.csv')
nltk.download('stopwords')
with st.expander('Example of dataset'):
    st.text('Our dataset is 8GB of JL-Data...')
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)   

with st.expander('Data Preparation'):
    st.text("Before Analyse to start we need to prepare our dataframe. To do this, we use several functions")
    d = {'Function': ["umlaut", "clean_tweet", "remove_rt", "remove_punkt", "re_umlaut"],
                         'Example' : ["Es wäre gut..", "@wue_reporter TOOOOOOORRRRR!!! #fcbayern","RT @aspd korrekt!", "Vorsicht!!! ich dachte, dass...", "Es waere gut.."],
                         'Result': ["Es waere gut..", "TOOOOOOORRRRR!!!", "@aspd korrekt!","Vorsicht ich dachte dass", "Es wäre gut.."]}
    table = pd.DataFrame(data=d)
    st.table(table)
    opt = st.selectbox("Word Cloud", (" ","Without Stopwords","With Stopwords"))
    if opt == " ":
        st.write(" ")
    elif opt == "Without Stopwords":
        text = ''
        for tweet in df['tweet']:
            text += ''.join(tweet.split(','))
        wordcloud = WordCloud(max_words=500, width=1500, height = 800, collocations=False).generate(text)
        fig = plt.figure(figsize=(20,20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        
    elif opt == "With Stopwords":
        
        stop_words = stopwords.words('german')    
        df['tweet'] = df['tweet'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        text = ''
        for tweet in df['tweet']:
            text += ''.join(tweet.split(','))
        wordcloud = WordCloud(max_words=500, width=1500, height = 800, collocations=False).generate(text)
        fig = plt.figure(figsize=(20,20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        
    if st.checkbox("Count of Tweets"):
        st.image("count.jpg", caption = "Count of Tweets per Party")



with st.expander("Prediction"):
    #stop_words = stopwords.words('german')    
    #df1['tweet'] = df['tweet'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))  
    X = df['tweet']
    y = df['party']
    vectorizer = TfidfVectorizer(max_features=3500, min_df=8, max_df=0.8)
    X = vectorizer.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    my_tags = df['party'].unique()
    
    new_tweet = st.text_area("Input a new Tweet for prediction")
    new_tweet = re_umlaut(remove_punkt(remove_rt(clean_tweet(umlaut(new_tweet.lower())))))
    
    if st.button("Prepare"):
        st.write(new_tweet)
  
    option = st.selectbox('ML Model', 
        ["Naive Bayes",
         "Linear Support Vector Machine", 
         "Logistic Regression"])
    
    if option == 'Naive Bayes':
        nb = MultinomialNB()
        nb.fit(X_train,y_train)
        nb_pred_res = nb.predict(X_test)
        if st.button("Predict"):
            nb_pred = nb.predict(vectorizer.transform([new_tweet]))
            st.write(nb_pred)
        
        if st.button("Evaluation"):
            st.text('Model Report:\n ' + classification_report(y_test, nb_pred_res, target_names=my_tags))
            
    
    elif option == 'Linear Support Vector Machine':
        sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=12, max_iter=5, tol=None)
        sgd.fit(X_train, y_train)
        sgd_pred_res = sgd.predict(X_test)
        
        if st.button("Predict"):
            sgd_pred = sgd.predict([new_tweet])
            st.write(sgd_pred)
            
        if st.button("Evaluation"):
            st.text('Model Report:\n ' + classification_report(y_test, sgd_pred_res, target_names=my_tags))
                  
    elif option == 'Logistic Regression':
        logreg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
        logreg.fit(X_train, y_train)
        lg_pred_res = logreg.predict(X_test)
        
        if st.button("Predict"):
            sgd_pred = logreg.predict([new_tweet])
            st.write(sgd_pred)
            
        if st.button("Evaluation"):
            st.text('Model Report:\n ' + classification_report(y_test, lg_pred_res, target_names=my_tags))

