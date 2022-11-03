#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#import time
import math
from urllib.request import urlopen
import json
import requests
#import plotly.graph_objects as go 
#import shap
#from sklearn.impute import SimpleImputer
#from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


def get_response(url):
    response = requests.post(url)
    print(response)
    return response.json()

#Chargement des données
df = pd.read_csv('app_test.csv')       



#Chargement du modèle
model = pickle.load(open('lgbm.pkl', 'rb'))

    #######################################
    # SIDEBAR
    #######################################

logo_image = "logo_oc.png"
shap_image = "Shap_features_importances.png"
stage_image = "stage.png"

with st.sidebar:
 st.header("💰 Client analysis")

 st.write("## Client ID")
 id_list = df["SK_ID_CURR"].tolist()
 id_client = st.selectbox(
            "Select Customer ID", id_list)

 st.write("## Actions")
 show_credit_decision = st.checkbox("View credit decision")
 shap_general = st.checkbox("View SHAP features importances")
 show_client_details = st.checkbox("View cliend info")

            

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    #Titre principal

html_temp = """
    <div style="background-color: gray; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard - Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Credit decision support for customer relationship managers</p>
    """
st.markdown(html_temp, unsafe_allow_html=True)

with st.expander("What is this app for?"):
        st.write("This app is used to predict the capacity of the client to repay the loan") 
        st.image(logo_image)


    #Afficher l'ID Client sélectionné
st.write("Select client ID :", id_client)



if (show_client_details):
    df_client = df[df['SK_ID_CURR'] == id_client]
    total_income = df_client['AMT_INCOME_TOTAL']
    duration_imployed = - df_client['DAYS_EMPLOYED']
    st.write('### Total income of the client: ', total_income)
    st.write('Avarage total income of the bank clients: ', 168797.91)
    st.write('### Working experience of the client in days: ', int(duration_imployed))
    st.write('‍General distribution of the working experience among the clients of the bank')
    st.image('stage.png')
    

    


    
        #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------

if (show_credit_decision):
    st.header('‍⚖️ Scoring et décision du modèle')

            #Appel de l'API : 

    API_url = "https://scoringmodelopen.herokuapp.com/predict?id_client" + str(id_client)
    #API_url = "https://scoringmodelopen.herokuapp.com/"
    json_url = get_response(API_url)
    st.write("## Json {}".format(json_url))
    API_data = json_url
    
    classe_predite = API_data['prediction']
    if classe_predite == 1:
        decision = 'Mauvais prospect (Crédit Refusé)'
        st.write(decision)
    else:
        decision = 'Bon prospect (Crédit Accordé)'
        st.write(decision)

        


        #-------------------------------------------------------
        # Afficher la feature importance globale
        #-------------------------------------------------------

if (shap_general):
    st.header('‍Feature importance (SHAP)')
    st.image('Shap_features_importances.png')
    

    
#streamlit run streamlit_app.py