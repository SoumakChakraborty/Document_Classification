import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from flask import Flask,request,app,jsonify,render_template
import PyPDF2
import os
import shutil
import re
from pathlib import Path
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app=Flask(__name__)
def extract_text_pdf(pth,file_name,q):
    #print(pth,file_name)
    file=pth+'\\'+file_name
    #print(file)
    pdf=PyPDF2.PdfReader(file)
    pg=pdf.getPage(0)
    ss=pg.extract_text()
    '''ss=ss.lower()
    a=ss.find('abstract')
    if ss.find('index terms')!=-1:
        b=ss.find('index terms')
    else:
        b=ss.find('keywords')
    extracted_text=ss[a:b]'''
    extracted_text=re.sub('\n',' ',ss)
    extracted_text=re.sub('[^A-Za-z0-9,.;]',' ',extracted_text)
    extracted_text=re.sub('\s+',' ',extracted_text)
    q.append(extracted_text)
    #return extracted_text

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_t=tf.keras.models.load_model('./best_model/check')
    #data1=request.form.get('data1')
    #data2=request.form.get('data2')
    print(request.files)
    data1=request.files['data1']
    data2=request.files['data2']
    if Path('./temp_data').exists()==False:
        os.mkdir('./temp_data')
    data1.save('./temp_data/f1.csv')
    data2.save('./temp_data/f2.csv')
    q=[]
    th1=threading.Thread(target=extract_text_pdf,args=('./temp_data','f1.csv',q))
    th2=threading.Thread(target=extract_text_pdf,args=('./temp_data','f2.csv',q))
    #t1=extract_text_pdf('./temp_data','f1.csv')
    #t2=extract_text_pdf('./temp_data','f2.csv')
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    #ttx1=[t1]
    #ttx2=[t2]
    ttx1=[q[0]]
    ttx2=[q[1]]
    #print(ex_text)
    #ttx1=ex_text1[0]
    #ttx2=ex_text2[0]
    shutil.rmtree('./temp_data')
    emb1=model.encode(ttx1)
    emb2=model.encode(ttx2)
    emb1=emb1[0]
    emb2=emb2[0]
    dist=1-cosine(emb1,emb2)
    d=np.array([dist])
    er=emb1.copy()
    er=np.append(er,emb2)
    er=np.append(er,d)
    er=er.reshape(1,er.shape[0],1)
    res=np.where(model_t.predict(er)[0][0]>=.5,1,0)
    outp=''
    if res==1:
        outp='Same document with probability {}'.format(str(model_t.predict(er)[0][0]))
    else:
        outp='Different document with probability {}'.format(str(1-model_t.predict(er)[0][0]))
    return render_template('index.html',prediction="Prediction:{}".format(outp))

if __name__=='__main__':
    app.run(host='0.0.0.0')