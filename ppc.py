import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas_profiling
import json
import glob
import string
import json_lines
from tqdm.notebook import tqdm
from pprint import pprint
import re
from string import punctuation
import preprocessor as p
from nltk.stem import *
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialN

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

# Extract text of tweet and name of party !!!Not for streamlit!!!
#     party = []
#     files = glob.glob('data/*.jl')
#     for file in tqdm(files):
#         with json_lines.open(file) as jl:
#             for item in jl:
#                 if 'response' in item:
#                     if 'data' in item['response']:
#                         for i in item['response']['data']:
#                             party.append({'party': item['account_data']['Partei'], 'tweet': i['text']})
#                             break
#     df = pd.DataFrame(party, columns = ['party', 'tweet'])

df = pd.read_csv('test.csv', on_bad_lines='skip')
###########define some functions
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
    text = re.sub(r'[^\w\s]','',text)   
    return text

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else '' for i in text])

def re_umlaut(text):
    text = text.replace('aeqwe', 'ä')
    text = text.replace('oeqwe', 'ö')
    text = text.replace('ueqwe', 'ü') 
    text = text.replace('ssqwe', 'ß')
    return text
 
###data prepare & save df in csv
# prep_text = [re_umlaut(remove_numbers(remove_punkt(remove_rt(clean_tweet(umlaut(text.lower())))))) for text in df['tweet']]
# df['tweet_prep'] = prep_text
# df.to_csv('test.csv', index=False, columns = ['party', 'tweet', 'tweet_prep'])
# ###remove NaN rows from Dataset
# df = pd.read_csv('test.csv')
# df1 = df.dropna(thresh=3)
# df1.to_csv('test.csv', index=False, columns = ['party', 'tweet', 'tweet_prep'])

### datei einlesen
#df = pd.read_csv('test.csv')
import nltk
nltk.download('stopwords')
with st.expander('Example of dataset'):
    st.text('Our dataset is 8GB of JL-Data...')
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)   

with st.expander('Data Preparation'):
    st.text("Before Analyse to start we need to prepare our dataframe. To do this, we use several functions")
    d = {'Function': ["umlaut", "clean_tweet", "remove_rt", "remove_punkt","remove_numbers", "re_umlaut"],
                         'Example' : ["Es wäre gut..", "@wue_reporter TOOOOOOORRRRR!!! #fcbayern","RT @aspd korrekt!", "Vorsicht!!! ich dachte, dass...", "Es kostet 400000 Euro", "Es waere gut.."],
                         'Result': ["Es waere gut..", "TOOOOOOORRRRR!!!", "@aspd korrekt!","Vorsicht ich dachte dass", "Es kostet  Euro", "Es wäre gut.."]}
    table = pd.DataFrame(data=d)
    st.table(table)
    if st.button("Show me result"):
        st.image("preparation.jpg")
    opt = st.selectbox("Word Cloud", (" ","Without Stopwords","With Stopwords"))
    if opt == " ":
        st.write(" ")
    elif opt == "Without Stopwords":
        text = ''
        for tweet in df['tweet_prep']:
            text += ''.join(tweet.split(','))
        wordcloud = WordCloud(max_words=500, width=1500, height = 800, collocations=False).generate(text)
        fig = plt.figure(figsize=(20,20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        
    elif opt == "With Stopwords":
        
        stop_words = stopwords.words('german')    
        df['tweet_prep'] = df['tweet_prep'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        text = ''
        for tweet in df['tweet_prep']:
            text += ''.join(tweet.split(','))
        wordcloud = WordCloud(max_words=500, width=1500, height = 800, collocations=False).generate(text)
        fig = plt.figure(figsize=(20,20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        
    if st.checkbox("Count of Tweets"):
        fig1 = plt.figure(figsize=(8,6))
        df.groupby('party').tweet_prep.count().plot.bar(ylim=0)
        st.pyplot(fig1)
    

with st.expander("Prediction"):
    stop_words = stopwords.words('german')    
    x = df['tweet_prep']
    y = df['party']
    vectorizer = TfidfVectorizer (max_features=3000, min_df=5, max_df=0.7)
    x = vectorizer.fit_transform(x).toarray()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    my_tags = df['party'].unique()
    
    new_tweet = st.text_area("Input a new Tweet for prediction")
    new_tweet = re_umlaut(remove_numbers(remove_punkt(remove_rt(clean_tweet(umlaut(new_tweet.lower()))))))
    
    if st.button("Prepare"):
        st.write(new_tweet)
    
    option = st.selectbox('ML Model', 
        ["Naive Bayes",
         "Linear Support Vector Machine", 
         "Logistic Regression","Test"])
    
    if option == 'Naive Bayes':
        nb = MultinomialNB()
        nb.fit(x_train,y_train)
        nb_pred_res = nb.predict(x_test)
        if st.button("Predict"):
            nb_pred = nb.predict(vectorizer.transform([new_tweet]))
            st.write(nb_pred)
        
        if st.button("Evaluation"):
            #st.write('Accuracy %s' % accuracy_score(y_test, nb_pred_res))
            st.text('Model Report:\n ' + classification_report(y_test, nb_pred_res, target_names=my_tags))
            
    
    elif option == 'Linear Support Vector Machine':
        sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=12, max_iter=5, tol=None)
        sgd.fit(x_train, y_train)
        sgd_pred_res = sgd.predict(x_test)
        
        if st.button("Predict"):
            sgd_pred = sgd.predict([new_tweet])
            st.write(sgd_pred)
            
        if st.button("Evaluation"):
            #st.write('accuracy %s' % accuracy_score(y_test, sgd_pred_res))
            st.text('Model Report:\n ' + classification_report(y_test, sgd_pred_res, target_names=my_tags))
                  
    elif option == 'Logistic Regression':
        logreg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
        logreg.fit(x_train, y_train)
        lg_pred_res = logreg.predict(x_test)
        
        if st.button("Predict"):
            sgd_pred = logreg.predict([new_tweet])
            st.write(sgd_pred)
            
        if st.button("Evaluation"):
            #st.write('accuracy %s' % accuracy_score(y_test, lg_pred_res))
            st.text('Model Report:\n ' + classification_report(y_test, lg_pred_res, target_names=my_tags))
        
        

