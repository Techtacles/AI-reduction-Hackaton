from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
#import os
#from pathlib import Path
app = Flask(__name__)
#dire='C:\\Users\\USER\\Desktop\\NEW DEPLOYMENT\\model.pkl'

modell=pickle.load(open("model.pkl","rb"))
scaled=pickle.load(open("scaled.pkl","rb"))
encoder=pickle.load(open("encoder.pkl","rb"))



@app.route('/')
def home():
	return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
	make=request.form["make"]
	model=request.form["model"]
	transmission=request.form["transmission"]
	engine=request.form["engine"]
	cylinders=request.form["cylinders"]
	fueltype=request.form["fuelType"]
	mpg=request.form["mpg"]
	cat=[[make,model,transmission,fueltype]]
	cat_encoded=encoder.transform(cat)

	num=[[engine,cylinders,mpg]]
	num_encoded=scaled.transform(num)
	final=np.hstack(([cat_encoded,num_encoded]))
	prediction=modell.predict(final)
	if prediction[0]>251:
	  return render_template("index.html",prediction_text=f"Your vehicle will emit  {np.round(prediction[0],2)} grams per kilometer to the atmosphere.This emits too much to the atmosphere")
	else:
   	  return render_template("index.html",prediction_text=f"Your vehicle emits {np.round(prediction[0],2)} grams per kilometer. This is a good range hence, importation of {make},{model} of engine {engine} will be granted. ")

if __name__ == '__main__':
    app.run(debug=False)