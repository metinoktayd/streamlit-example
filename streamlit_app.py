import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #karar ağacı
from sklearn.feature_extraction.text import CountVectorizer #kelimeleri vektöre çevirir
from sklearn.model_selection import train_test_split
import string
import streamlit as st
import sqlite3
import datetime

zaman=str(datetime.now())

conn=sqlite3.connect("comment.sqlite3")
c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS tests(yorum TEXT,sonuc TEXT,zaman TEXT)")
conn.commit()

df=pd.read_csv('comment.csv', on_bad_lines="skip",delimiter=";")

def temizle(sutun):
    semboller=string.punctuation
    sutun=sutun.lower()
    for sembol in semboller:
        sutun=sutun.replace(sembol," ")
    stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
    for stopwords in stopwords:
        s=" "+stopwords+" "
        sutun=sutun.replace(s," ")
    sutun=sutun.replace("  "," ")
    return sutun
df['Metin']=df['Metin'].apply(temizle)

cv=CountVectorizer(max_features=150)
X=cv.fit_transform(df['Metin']).toarray()
y=df['Durum']

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=42)

rf=RandomForestClassifier()
model=rf.fit(X,y)
model.score(X,y)

y=st.text_area("Yorum Metinini Giriniz")
btn=st.button("Yorumu Kategorilendir")
if btn:
    rf = RandomForestClassifier()
    model = rf.fit(x_train, y_train)
    score=model.score(x_test, y_test)

    tahmin = cv.transform(np.array([y])).toarray()
    kategori = {
        0: "Olumsuz",
        1: "Olumlu",
        2: "Nötr"
    }
    sonuc = model.predict(tahmin)
    s=kategori.get(sonuc[0])
    st.subheader(s)
    st.write("Model Skoru",score)
    c.execute("INSERT INTO yorumlar VALUES(?,?,?)",(y,s,zaman))
    conn.commit()

c.execute("SELECT * FROM tests")
tests=c.fetchall()
st.table(tests)
