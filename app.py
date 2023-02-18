# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
classifier = pickle.load(open('diabetes-prediction-rfc-model.pkl', 'rb'))

model = pickle.load(open('heart-disease-prediction-knn-model.pkl', 'rb'))

modelk = pickle.load(open('Kidney.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/templates/diabetes.html')
def diabetes():
    return render_template('diabetes.html')

@app.route('/templates/diabetes.html/', methods=['POST'])
def predict1():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

@app.route('/Heart')
def heart():
    return render_template('main.html')


@app.route('/predict2', methods=['GET','POST'])
def predict2():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('Heart_result.html', prediction=my_prediction)



@app.route('/templates/index4.html',methods=['GET'])
def Home():
    return render_template('index4.html')

@app.route("/templates/index4.html", methods=['POST'])
def predict3():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = modelk.predict(values)

        return render_template('result4.html', prediction=prediction)


        

if __name__ == '__main__':
	app.run(debug=True)