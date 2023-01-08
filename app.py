import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from flask import Flask,request,app,jsonify,render_template
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_t=tf.keras.models.load_model('./best_model/check')
    data=request.json['data']
    ttx1=[data[0]]
    ttx2=[data[1]]
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
    return jsonify(str(res))

if __name__=='__main__':
    app.run(debug=True)