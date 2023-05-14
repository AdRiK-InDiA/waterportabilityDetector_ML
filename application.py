import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
model=pickle.load(open(f'{os.getcwd()}/notebook/articrafts/model.pkl','rb'))
standard_scaler=pickle.load(open(f'{os.getcwd()}/notebook/articrafts/scaler.pkl','rb'))
## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST']) # type: ignore
def predict_datapoint():
    if request.method=='POST':
        ph=float(request.form.get('ph')) # type: ignore
        hardness= float(request.form.get('hardness')) # type: ignore
        tds = float(request.form.get('tds'))*1000.0 # type: ignore
        sulphate = float(request.form.get('sulphate')) # type: ignore
        chloramines = float(request.form.get('chloramines')) # type: ignore
        conductivity = float(request.form.get('conductivity')) # type: ignore
        toc = float(request.form.get('toc')) # type: ignore
        Trihalomethanes = float(request.form.get('Trihalomethanes')) # type: ignore
        Turbidity = float(request.form.get('Turbidity')) # type: ignore
        

        new_data_scaled=standard_scaler.transform([[ph,hardness,tds,chloramines,sulphate,conductivity,toc,Trihalomethanes,Turbidity]])
        result=model.predict(new_data_scaled)[0]
        
        if result==0:
            result = 'unportable'
        else :
            result = 'portable'

        return render_template('results.html',prediction=result)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=5500)
