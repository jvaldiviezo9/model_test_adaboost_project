import streamlit as st
import pickle as pkl
import sklearn as sk
import pandas as pd
import numpy as np

st.title("Project ML Adaboost")

st.header("Exploracion de datos")

st.header("Visualizacion")

col1, col2, col3 = st.columns(3)

data_arr = np.zeros(12)

with col1:
  
  # variables socioeconomicas
  
  st.header("Variables Socioeconomicas")
  
  gender = st.selectbox("Genero", ("M","F"))
  if gender == "M":
    data_arr[1] = 1
  else:
    data_arr[1] = 0
  
  age = st.slider("Edad", 18, 99)
  data_arr[2] = age
  
  country = st.selectbox("Pais", ("Francia", "Alemania", "España", "Otro"))
  
  if country == "Francia":
    data_arr[8] = 1
  elif country == "Alemania":
    data_arr[9] = 1
  elif country == "España":
    data_arr[10] = 1
  else:
    pass

  
with col2:
  
  # caracteristicas del trabajo
  
  tenure = st.slider("Edad", 0, 10)
  data_arr[3] = tenure
  
  salary = st.slider("Edad", 0, 250000)
  data_arr[11] = tenure
  
with col3:
  # banco
  
  creditscore = st.slider("Score Crediticio", 300, 900)
  data_arr[0] = creditscore
  
  balance = st.slider("Balance", 0, 250000)
  data_arr[4] = balance
  
  numofproducts = st.slider("Productos", 0,4)
  data_arr[5] = numofproducts
  
  isactive = st.selectbox("Activo", ("Si", "No"))
  if isactive == "Si":
    data_arr[7] = 1
  else:
    data_arr[7] = 0

  
  creditcard = st.selectbox("Tarjeta de Credito", ("Si", "No"))
  if creditcard == "Si":
    data_arr[6] = 1
  else:
    data_arr[6] = 0
    
    
data_arr =  np.expand_dims(data_arr, axis=0)

model = pkl.load(open("ada_estimator.pkl", "rb"))

if st.button("Predecir"):
  pred = model.predict_proba(data_arr)
  
  st.text(f"Usted tiene una probabilidad de { round(pred[0][1] * 100, 1)} % de abandonar el banco")
else:
  st.text("Click en Predecir")

