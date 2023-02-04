import requests
import csv
import os
import io
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.message import EmailMessage
from email import encoders
import mlflow

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px 
import plotly.graph_objects as go

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://host.docker.internal:9000"


# remote_server_uri = "http://localhost:5001"# set to your server URI
remote_server_uri = "http://mlflow_server:5000"# set to your server URI

mlflow.set_tracking_uri(remote_server_uri)
# model_name = "cbnGDP"
model_version = 1
run_ID = '24fb317b6ed24d689bbf27ff54488466'
model_name = "newGDPmodel"
model = mlflow.load_model(model_uri=f"models:/{model_name}/{model_version}")

st.set_page_config(layout="wide")
st.sidebar.markdown("## Enaira Anomaly Detection Model")
app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_Fraud'])
if app_mode=='Home': 
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:green;padding:6px"> 
    <h3 style ="color:White;text-align:center;">Enaira Anomaly Detection Model</h3> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    # if app_mode=='Home': 
    st.title('Predict Fraudulent/Non Fraudlent Enaira Users') 
    st.markdown('Dataset :') 
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dd=df[['SOURCE_WALLET_GUID', 'date', 'AMOUNT_Cleaned', 'TIER_LEVEL', 'MERCHANT_LABEL',
       'CURRENT_STATE', 'KYC_STATUS', 'MailGroup', 'bvn_flag', 'month', 'day',
       'hour', 'minute', 'merchant_label_', 'tier_level_', 'current_state_',
       'MailGroup_', 'bvn_flag_', 'kyc_status_',  'Fraud_Flag']].copy()
        

        
        dd_s=dd.head(10)
            # style
        th_props = [
          ('font-size', '18px'),
          ('text-align', 'center'),
          ('font-weight', 'bold'),
          ('color', '#6d6d6d'),
          ('background-color', '#f7ffff')
          ]

        td_props = [
          ('font-size', '15px')
          ]

        styles = [
          dict(selector="th", props=th_props),
          dict(selector="td", props=td_props)
          ]

        # table
        df2=dd_s.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
        st.table(df2)

        st.line_chart(dd[["date", "AMOUNT_Cleaned"]].set_index("date"),width=100,height=700)
elif app_mode == 'Predict_Fraud':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        def make_predictions(df):
            ##load model
            amt_model=model
            #amt_model=pickle.load(open ('en_amount_model_streamlit.pkl','rb'))
            ##load label encoder
            file = open("enc_v1_amt_23.obj",'rb')
            enc_loaded = pickle.load(file)

            df_p=df[['SOURCE_WALLET_GUID', 'AMOUNT_Cleaned', 'TIER_LEVEL', 'MERCHANT_LABEL',
                     'CURRENT_STATE', 'KYC_STATUS', 'MailGroup', 'bvn_flag', 'month', 'day',
                     'hour', 'minute']]
            df_p[['merchant_label_','tier_level_','current_state_','MailGroup_','bvn_flag_','kyc_status_']]=df_p[['MERCHANT_LABEL', 'TIER_LEVEL','CURRENT_STATE', 'MailGroup', 'bvn_flag', 'KYC_STATUS']]
  
            df_u=df_p[['merchant_label_','tier_level_','current_state_','MailGroup_','bvn_flag_','kyc_status_','month', 'day', 'hour','minute', 'AMOUNT_Cleaned']]
            col=['merchant_label_','tier_level_','current_state_','MailGroup_','bvn_flag_','kyc_status_']

            df_u[col]= enc_loaded.transform(df_u[col])

            predss= amt_model.predict(df_u)

            final_pred= pd.DataFrame({'SOURCE_WALLET_GUID':  df.SOURCE_WALLET_GUID,
                                       'date': df.date,
                                       'bvn_flag':df.bvn_flag,
                                       'MailGroup':df.MailGroup,
                                       'KYC_STATUS':df.KYC_STATUS,
                                       'TIER_LEVEL':df.TIER_LEVEL,
                                       'MERCHANT_LABEL':df.MERCHANT_LABEL,
                                       'CURRENT_STATE':df.CURRENT_STATE,
                                       'AMOUNT':df.AMOUNT_Cleaned,
                                       'day':df.day,
                                       'month':df.month,
                                       'hour':df.hour,
                                       'minute':df.minute,
                                       'Prediction':predss})

            th_props = [
              ('font-size', '18px'),
              ('text-align', 'center'),
              ('font-weight', 'bold'),
              ('color', '#6d6d6d'),
              ('background-color', '#f7ffff')
              ]

            td_props = [
              ('font-size', '15px')
              ]

            styles = [
              dict(selector="th", props=th_props),
              dict(selector="td", props=td_props)
              ]

            final_pred=final_pred[['SOURCE_WALLET_GUID','date','TIER_LEVEL','MERCHANT_LABEL','CURRENT_STATE','KYC_STATUS',
             'month', 'day', 'hour', 'minute','MailGroup','bvn_flag', 'AMOUNT', 'Prediction']]
            final_pred=final_pred.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
            final_pred.set_properties(subset=['Prediction'],**{'background-color': 'red'})
            return final_pred 
        def send_email(user,pwd,subject):
            df_s=make_predictions(df)
            print(df_s)
            #df_s.set_properties(**{'background-color': 'red'}, subset=['Prediction'])
            try:
                df_html=df_s.hide_index().render()

                recipients=["foyelami@bluechiptech.biz"]
#                                'raolaniyan@cbn.gov.ng',
#                                                     'amuwais@cbn.gov.ng',
#                                                     'tbadekayero@cbn.gov.ng',
#                                                     'uiisiyaku@cbn.gov.ng',
#                                                     'edetim@cbn.gov.ng',
#                                                     'aaliyu5@cbn.gov.ng',
#                                                     'raolaniyan@cbn.gov.ng',
#                                                     'bnoyekanmi@cbn.gov.ng']

                msg =MIMEMultipart('alternative')

                msg['Subject']=subject
                msg['From']=user
                msg['To']=",".join(recipients)

                html= """\
                <html>
                    <head>
                    </head>
                      <link rel='stylesheet' href='http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css'>
                       <body>
                           <p>
                           <br>Hello Ops Team</br>

                            <br>
                           The following users have been flagged for making suspicious transactions: please look into this
                           </br>
                           </br>

                           </br> 
                           </p>
                       </body>
                </html>
                """
                #html = "df_html".join((df_html,message_style))
                html += df_html
                part2 = MIMEText(html.encode('utf-8'),'html','utf-8')

                #msg.attach(dfPart2)
                msg.attach(part2)
                #df3=df_s.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
                #st.table(df3)
                st.write(df_s)


                server=smtplib.SMTP("smtp.office365.com",587)
                server.starttls()
                server.login(user,pwd)

                server.sendmail(user, recipients, msg.as_string())
                server.close()
            

                print("Mail Sent!")

            except Exception as e:
                print(str(e))
                print("Failed to send mail")
    def main():
        send_email("foyelami@bluechiptech.biz","Goldfinch22","DataOps: Anonamly Detection System  !!!")
        print('email sent successfully')
                
    if st.button("Make Predictions"):
        if __name__ == '__main__':main()
        st.success('Mail Sent Succesffuly')
    else:
        st.write('No attached Document')