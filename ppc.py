import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas_profiling
import json as json
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
from sklearn.metrics import confusion_matrix
import nltk
st.set_page_config(page_icon="⭐", page_title="Political Party Tweet Classification", layout="wide")
selected = option_menu(None, ["Home", "Dataset",  "Process", 'Live Demo'], 
    icons=['house', 'file-earmark-text', "cpu", 'collection-play'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
)
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
## HOME
if selected=="Home":
    st.markdown("<h1 style='text-align: center'>Political Party Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>My Goal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>The goal of my project is to classify tweets of german politicans by the political party of the author. However, i don't just want to research the politicians and categorize them manually, i want to use Machine Learning algorithms.</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>About me</h2>", unsafe_allow_html=True)
    st.image("https://scontent-muc2-1.xx.fbcdn.net/v/t31.18172-8/20451827_1559088087491677_5562632512013699296_o.jpg?_nc_cat=103&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=tGN8W7JAj50AX-aYzJP&_nc_ht=scontent-muc2-1.xx&oh=00_AT-jaPGEyKBWeZIvPeg1mUn-5pWgqttKyRWkFXW26WcXPw&oe=630EE21E")
    st.markdown("<h5 style='text-align: center'>Sergei Mezhonnov </h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center'>4th semester Wirtschaftsinformatik</h6>", unsafe_allow_html=True)
    st.markdown("<p>Hi! I'm Sergei and I'm a working student at Siemens with focus on Database and IT-Solutions. \
    This Project helped me to use theoretical knowledge in Data Science into practical way. \
    In my free time i enjoy going to the gym and suffer there..</p>", unsafe_allow_html=True)
        
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

df = pd.read_csv('result.csv')
nltk.download('stopwords')
if selected=="Dataset":
    st.markdown("<h1 style='text-align: center'>Twitter Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p>My dataset is a JSON file consisting of official tweets from members of the german parliament as of march 2021. Thus it includes tweets from CDU/CSU, SPD, Die Gruenen, Die Linken, AFD, etc. \n The main problem one will soon discover is...</p>", unsafe_allow_html=True)
    
    st.markdown("<h5>...my dataset is 8GB of JL-Data...</h5>", unsafe_allow_html=True)
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)
        
## PROCESS
if selected=="Process":
    st.markdown("<h1 style='text-align: center'>CRISP-DM</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>For my project we used the procedure model CRISP-DM, which includes the following steps:</p>", unsafe_allow_html=True)
    cola, colb, colc = st.columns(3)
    colb.image('crisp_dm.jpg')
    with st.expander("Business Understanding"):
        st.markdown("<p>Everybody knows Tweets. You can retweet a tweet or you can create a new one completly on your own.\
        There are almost no limits to what you can include in your tweet. You can use text, numbers and emojicons. \
        Despite the almost unlimited possibilites to write a tweet, one might use same patterns - like special emojis or syntax - over and over again. \
        Furthermore, members of some political parties tend to write more about special topics like 'football' and less about other topics like 'gardening'. \
        The interesting part is to find those exact patterns. Some are quite obvious and others are rather inconspicuous. \
        However, i do not need to find those patterns on my own and read all of the 5000 tweets, i will use KI-algorithms for this!</p>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        col3.markdown("<h6 style='text-align: center'>Olaf Scholz retweeting Tagesschau</h6>", unsafe_allow_html=True)
        col3.image('olaf_scholz_tweet.png')
        
        col4.markdown("<h6 style='text-align: center'>Susanne Henning-Welson tweeting a link</h6>", unsafe_allow_html=True)
        col4.image('susanne_henning_tweet.png')
        
    with st.expander("Data Understanding"):
        st.markdown("<h6 style='text-align: center'>tar.gz-File</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>It is inevitable for a good data analysis to understand the structure of the source data. \
        Therefore i needed to understand the structure of the given file. Within the compressed file i got where smaller seperate JSON-line files.</p>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center'>JSON-File</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Furthermore i needed to understand how a JSON file works and which particular information was included</p>", unsafe_allow_html=True) 
        st.write("Included important information:")
        st.write("- http status")
        st.write("- account name")
        st.write("- account data (e.g. account name and party")
        st.write("- response")
        #st.markdown("<ul style='text-align:center'><li>http status</li>\
        #<li>account data (e.g. account name and party)</li><li>response</li></ol>", unsafe_allow_html=True)
        st.write("In summary i got information about the content of the tweet, as well as the authors name and party")
        
        if st.checkbox("Count of Tweets"):
            st.image("count.jpg", caption = "Count of Tweets per Party")
            
    with st.expander("Data Preparation"):
        st.markdown("<br><h6 style='text-align: center'>Changes</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Before analysis starts we need to prepare my dataframe. To do this, we used several functions. Here are some examples:</p>", unsafe_allow_html=True) 
        st.write("- convert source data into csv")
        st.write("- reduce the amount of tweets")
        st.write("- transform ä,ö,ü")
        st.write("- remove unimportant information like 'time'")
        st.write("- remove @-mentions")
        st.write("- remove punctuation")
        st.write("- remove retweet information")
        #st.markdown("<ul style='text-align:center'><li>convert source data into csv</li>\
        #<li>reduce the amount of tweets</li><li>transform ä,ö,ü</li><li>remove unimportant information like 'time'</li><li>remove @-mentions</li>\
        #<li>remove punctuation</li><li>remove retweet information</li></ol>", unsafe_allow_html=True)
        st.markdown("<br><h6 style='text-align: center'>Differences</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>In the following table one can see the difference of a raw tweet and the tweet after preparation</p>", unsafe_allow_html=True) 
        d = {'Function': ["umlaut", "clean_tweet", "remove_rt", "remove_punkt", "re_umlaut"],
                             'Example' : ["Es wäre gut..", "@wue_reporter TOOOOOOORRRRR!!! #fcbayern","RT @aspd korrekt!", "Vorsicht!!! ich dachte, dass...", "Es waere gut.."],
                             'Result': ["Es waere gut..", "TOOOOOOORRRRR!!! fcbayern", "@aspd korrekt!","Vorsicht ich dachte dass", "Es wäre gut.."]}
        table = pd.DataFrame(data=d)
        st.table(table)
        
        st.markdown("<br><h6 style='text-align: center'>Stop words</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Another thing i tried is to use the stop words function, which removes unnecessary german words from a text. Unnecessary as such are for example pronouns. \n \
        In the following wordclouds one can see the most common words in my dataset including the stop words function as well as without the function.</p>", unsafe_allow_html=True) 
        opt = st.selectbox("Word Cloud", ("Please choose...","Without Stopwords Function","With Stopwords Function"))
        if opt == "Please choose...":
            st.write(" ")
        elif opt == "Without Stopwords Function":
            text = ''
            for tweet in df['tweet']:
                text += ''.join(tweet.split(','))
            wordcloud = WordCloud(max_words=500, width=1500, height = 800, collocations=False).generate(text)
            fig = plt.figure(figsize=(20,20))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)
        elif opt == "With Stopwords Function":
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
        st.markdown("<h6 style='text-align: center'>Count of tweets</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Please select the checkbox if interested in how many tweets per party are include in the dataframe after preparation</p>", unsafe_allow_html=True) 
            
        st.text("After the data preparation i really did understand this meme:")
        st.image("meme.png", caption="Meme from Gitbook")
        
    with st.expander("Modeling"):
        st.markdown("<p style='text-align: center'>Part of this process section is to decide which algorithms to use. I decided to use the following three: Naive Bayes, Linear Support Vector Machine, Logistic Regression.</p>", unsafe_allow_html=True) 
        ## NB
        st.markdown("<h6 style='text-align: center'>Naive Bayes</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>This supervised learning algorithm is based on applying the Bayes\' theorem with the assumption of conditional independence between every pair of features. \
        In my project we used a special form of nb: Naive Bayes multinominal (due to my multiclassification problem). \
        Despite some other simple classifiers nb can work really well on reallife data, which is one of the main resonse i choose this one. Apart from that it takes an appropiate amount of time to train.</p>", unsafe_allow_html=True) 
        col1, col2, col3 = st.columns(3)
        col2.markdown("<a style='text-align: center' href='https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html'>Click here for more information</a>", unsafe_allow_html=True)
        col2.image("https://miro.medium.com/max/1200/1*ZW1icngckaSkivS0hXduIQ.jpeg", caption="Naive Bayes binomial")

        ## SVM
        st.markdown("<h6 style='text-align: center'>Stochastic Gradient Descent</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>This algorithm is rather a way of training a classifier than a specific family of machine learning models. \
        In my case this optimization technique an algorithm uses a SVM. Key points for my decision to use this was the efficiency.</p>", unsafe_allow_html=True) 
        col4, col5, col6 = st.columns(3)
        col5.markdown("<a style='text-align: center' href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html'>Click here for more information</a>", unsafe_allow_html=True)
        col5.image("https://eloquentarduino.github.io/wp-content/uploads/2020/04/SGD.jpg")
        
        ## LR        
        st.markdown("<h6 style='text-align: center'>Logistic Regression</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>This algorithm uses different inputs to determine the output (in my use case political party). To achieve LR uses an equation.</p>", unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)
        col8.markdown("<a style='text-align: center' href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'>Click here for more information</a>", unsafe_allow_html=True)        
        col8.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_multinomial_thumb.png")
        
    with st.expander("Evaluation"):
        st.markdown("<p style='text-alighn: center'>Obviously training my KI wasn't a linear process. I needed to iterate over different tasks over and over again, because we gained more knowledge through each step and iteration.\
        Especially the evaluation phase encouraged us to go over the preparation and modeling phase a lot.</p>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center'>Key points</h6>", unsafe_allow_html=True)
        st.markdown("<ul style='text-align:center'><li>removing stopwords worsened my results</li><li>setting the amount of tweets per party to the same number didn't really improve the results</li><li>changing the training data and only focusing on particular members did improve our results a lot</li></ol>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>In summary i discovered a lot of setbacks and iterations with very bad accuracy. \
        However, the change that really did the trick was to not reduce the source data random or based on the party, but rather based on particular politicians.</p>", unsafe_allow_html=True)
        st.text("Sidenote: current evaluation data are shown in the live demo")
   
    with st.expander("Deployment"):
        st.markdown("<p style='text-alighn: center'>The deployment itself was quite smooth. However, i run some problems with the streamlit online version. The online version had problems with 'checking the health of the project' while the local version worked very well.</p>", unsafe_allow_html=True)  
if selected=="Live Demo":  
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
         "Stochastic Gradient Descent", 
         "Logistic Regression"])
    
    if option == 'Naive Bayes':
        nb = MultinomialNB()
        nb.fit(X_train,y_train)
        nb_pred_res = nb.predict(X_test)
        if st.button("Predict"):
            nb_pred = nb.predict(vectorizer.transform([new_tweet]))
            st.balloons()
            st.write(nb_pred)
        
        if st.button("Evaluation"):
            st.markdown("<h6>Key figures</h6>", unsafe_allow_html=True)
            st.markdown("<p>In the following report the most important figures are shown.</p>", unsafe_allow_html=True)
            st.text('Model Report:\n ' + classification_report(y_test, nb_pred_res, target_names=my_tags))
            st.markdown("<h6>Confusion Matrix</h6>", unsafe_allow_html=True)
            st.markdown("<p>To get a more detailed overview of the performance please take a look at this matrix.</p>", unsafe_allow_html=True)
            cf_matrix = confusion_matrix(y_test, nb_pred_res)
            data = pd.DataFrame(cf_matrix)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='index', inplace=False)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='columns', inplace=False)
            st.table(test)
            
    
    elif option == 'Stochastic Gradient Descent':
        sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=12, max_iter=5, tol=None)
        sgd.fit(X_train, y_train)
        sgd_pred_res = sgd.predict(X_test)
        
        if st.button("Predict"):
            sgd_pred = sgd.predict(vectorizer.transform([new_tweet]))
            st.write(sgd_pred)
            
        if st.button("Evaluation"):
            st.markdown("<h6>Key figures</h6>", unsafe_allow_html=True)
            st.markdown("<p>In the following report the most important figures are shown.</p>", unsafe_allow_html=True)
            st.text('Model Report:\n ' + classification_report(y_test, sgd_pred_res, target_names=my_tags))
            
            st.markdown("<h6>Confusion Matrix</h6>", unsafe_allow_html=True)
            st.markdown("<p>To get a more detailed overview of the performance please take a look at this matrix.</p>", unsafe_allow_html=True)
            cf_matrix = confusion_matrix(y_test, sgd_pred_res)
            data = pd.DataFrame(cf_matrix)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='index', inplace=False)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='columns', inplace=False)
            st.table(test)
                  
    elif option == 'Logistic Regression':
        logreg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
        logreg.fit(X_train, y_train)
        lg_pred_res = logreg.predict(X_test)
        
        if st.button("Predict"):
            lg_pred = logreg.predict(vectorizer.transform([new_tweet]))
            st.write(lg_pred)
            
        if st.button("Evaluation"):
            st.markdown("<h6>Key figures</h6>", unsafe_allow_html=True)
            st.markdown("<p>In the following report the most important figures are shown.</p>", unsafe_allow_html=True)
            st.text('Model Report:\n ' + classification_report(y_test, lg_pred_res, target_names=my_tags))
            st.markdown("<h6>Confusion Matrix</h6>", unsafe_allow_html=True)
            st.markdown("<p>To get a more detailed overview of the performance please take a look at this matrix.</p>", unsafe_allow_html=True)
            cf_matrix = confusion_matrix(y_test, lg_pred_res)
            data = pd.DataFrame(cf_matrix)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='index', inplace=False)
            test = data.set_axis(['Bündnis 90/Die Grünen', 'SPD', 'AfD', 'Die Linke', 'FDP', 'CSU', 'CDU', 'Fraktionslos'], axis='columns', inplace=False)
            st.table(test)
