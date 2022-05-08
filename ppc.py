import streamlit as st
import pandas as pd
import numpy as np
import json
import json_lines

#Present your team
st.title('Political Party Classification')
with st.expander("Our Team:"):
    col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Jan Amend"}</h1>', unsafe_allow_html=True)
    st.image("https://www.sprit-plus.de/sixcms/media.php/5172/thumbnails/Mann_Fragezeichen.jpg.31563300.jpg")

with col2:
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Jana Adler"}</h1>', unsafe_allow_html=True)
    st.image("https://www.sprit-plus.de/sixcms/media.php/5172/thumbnails/Mann_Fragezeichen.jpg.31563300.jpg")

with col3:
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Sergei Mezhonnov"}</h1>', unsafe_allow_html=True)
    st.image("https://scontent-frt3-2.xx.fbcdn.net/v/t31.18172-8/20451827_1559088087491677_5562632512013699296_o.jpg?_nc_cat=103&ccb=1-5&_nc_sid=09cbfe&_nc_eui2=AeEdsPUBJzyPOZFs7ylgC0hgD7d7OvNp0kMPt3s682nSQ3wLFAZVTOrmLSOyHzD2mTt9LJKE7h7ZjiAONvU_-HJN&_nc_ohc=6ihYdGJVRsYAX_5hGRe&_nc_ht=scontent-frt3-2.xx&oh=00_AT893UIInBTyBmNpf071lnSEZBuEZS3igGFkVQEABTd9rg&oe=6294469E")

#present your project
with st.expander("About Project:"):
    st.text('  The goal of our project is to classify Tweets by the political party\n of the author')
#show one element of your dataset
with st.expander('Example of dataset'):
    st.write('Example')
    with json_lines.open('data/johannesvogel.jl') as jl:
        for item in jl:
            for tweet in item['response']['data']:
                st.json(tweet)
                break
            break

if st.button("What is a classificator?"):
    st.image("https://miro.medium.com/max/1400/1*R6Rbcks-pGO0SkhCINrP0g.png")