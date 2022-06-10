#import Flask 
from flask import Flask
import numpy as np
import joblib
from flask import Flask, render_template, request

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        kota = request.form.get('kota')
        kec = request.form.get('kec')
        trw = request.form.get('trw')
        trt = request.form.get('trt')
        tkk = request.form.get('tkk')
     
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(kota, kec, trw, trt, tkk)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(kota, kec, trw, trt, tkk):
    
    #keep all inputs in array
    test_data = [kota, kec, trw, trt, tkk]
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    #open file
    file = open("logreg_model.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    
    pass

if __name__ == '__main__':
    app.run(debug=True)