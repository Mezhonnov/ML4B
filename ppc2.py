import pandas as pd
import time
import random as rd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

st.set_page_config(page_title="für Meeting", layout="wide") 
st.title("Für Meeting")
df = pd.read_csv('tabelle.csv')

del df['Enthusiasm']
del df['Pitch']
del df['Meeting Date']
df['Country'] = ['US', 'DE', 'US', 'FR', 'DE', 'DE', 'IT', 'US', 'DE', 'IT']
np.random.seed(42)
df['Costs'] = np.random.randint(10000,100000, size=len(df))
df1 = df.groupby('Country')
with st.expander('Hier können Sie die Tabelle anschauen'):
    shows = df
    gb = GridOptionsBuilder.from_dataframe(shows)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()

    AgGrid(shows, gridOptions=gridOptions, enable_enterprise_modules=True)

with st.expander('Daten gruppieren und visualisieren'):
    st.text('Z.B. Wir wollen Kosten pro Land gruppieren')
    if st.button('Ich will Daten gruppieren'):
        st.dataframe(df1.sum())
        
    if st.button('Ich will Daten visualisieren'):
        time.sleep(1)
        st.bar_chart(df1.sum())