# Import Library
import komputasi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

col1, col2 = st.columns(2)
col1.title('IngfoIn')
col1.write('IngfoIn adalah aplikasi yang bisa membantu Anda dalam menentukan strategi marketing terbaik di toko atau mini market Anda.')

image = Image.open('images/orbit.jpg')
col2.image(image)

image = Image.open('images/minimarket.jpg')
st.image(image)
link = "Contoh dataset bisa didownload [di sini](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset?datasetId=877335&sortBy=voteCount)"
st.markdown(link,unsafe_allow_html=True)
df = None
dataset_file = st.file_uploader("Upload Dataset Anda", type=['csv'])
try:
    df = pd.read_csv(dataset_file)
except:
    st.warning('Mohon upload dataset Anda!')
    st.stop()

# Get Cols Names
pembeli = df.columns[0]
tanggal = df.columns[1]
produk = df.columns[2]

# Data Mining
df = komputasi.data_summary(df, pembeli, tanggal, produk)

# MBA using Apriori
komputasi.MBA(df, pembeli, produk)