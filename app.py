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
	return render_template("index2.html")

@app.route('/predict',methods=['POST'])
def predict():
	make=request.form["make"].upper()
	model=request.form["model"].upper()
	transmission=request.form["transmission"].upper()
	engine=request.form["engine"]
	cylinders=request.form["cylinders"]
	fueltype=request.form["fueltype"].upper()
	mpg=request.form["mpg"]
	cat=[[make,model,transmission,fueltype]]
	cat_encoded=encoder.transform(cat)

	num=[[engine,cylinders,mpg]]
	num_encoded=scaled.transform(num)
	final=np.hstack(([cat_encoded,num_encoded]))
	prediction=modell.predict(final)
	return render_template("index2.html",prediction_text=f"Your vehicle will emit  {prediction[0]} grams per kilometer to the atmosphere")
   	

if __name__ == '__main__':
    app.run(debug=False)