print("Starting App...")

from flask import Flask, request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

import os
import os
print("Current Working Directory:", os.getcwd())
print("Files in this directory:", os.listdir())

base_dir = os.path.abspath(os.path.dirname(__file__))

# This joins that path with the 'templates' folder name
template_dir = os.path.join(base_dir, 'templates')
# Update this line
application = Flask(__name__, template_folder='src/templates')
app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        import src.pipeline.predict_pipeline as pp
        print("FILE LOCATION:", pp.__file__)
        print("CLASS CONTENT:", dir(CustomData))
        data = CustomData(
            sqft_living = float(request.form.get('sqft_living') or 0),
            sqft_lot = float(request.form.get('sqft_lot') or 0),
            date = request.form.get('date'),
            bedrooms = int(request.form.get('bedrooms') or 0),
            bathrooms = float(request.form.get('bathrooms') or 0),
            sqft_basement = float(request.form.get('sqft_basement') or 0),
            sqft_above = float(request.form.get('sqft_above') or 0),
            condition = int(request.form.get('condition') or 0),
            view = int(request.form.get('view') or 0),
            waterfront = int(request.form.get('waterfront') or 0),
            floors = float(request.form.get('floors') or 0),
            statezip = request.form.get('statezip'), # This is a string
            yr_renovated = int(request.form.get('yr_renovated') or 0),
            yr_built = int(request.form.get('yr_built') or 0)
        ) 
        import inspect
        print("DEBUG: CustomData arguments:", inspect.signature(CustomData.__init__))

        import inspect
        print("DEBUG Signature:", inspect.signature(CustomData.__init__))
        print("ALL FORM DATA:", request.form)

        # 3. Processing
        print(">>> Step 2: Converting to DataFrame")
        pred_df = data.get_data_as_data_frame()
        
        print(">>> Step 3: Loading Pipeline and Model")
        predict_pipeline = PredictPipeline()
        
        print(">>> Step 4: Running Prediction")
        results = predict_pipeline.predict(pred_df)
        
        print(">>> Step 5: Prediction Finished!")
        import numpy as np

# Convert the log-prediction back to actual dollars
        predicted_price = np.expm1(results[0])
        
        # Round it to 2 decimal places so it looks professional
        formatted_price = round(float(predicted_price), 2)
        
        return render_template('home.html', results=formatted_price)
        #return render_template('home.html', results=results[0])

    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True,threaded=True) 

print("Imports finished, starting server...")    