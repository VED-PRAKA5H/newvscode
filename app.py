from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)

# Set debug mode for development
app = application
app.debug = True

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Render the home page when accessed via GET
    else:
        # Collect data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),  # Corrected mapping
            writing_score=int(request.form.get('writing_score'))   # Corrected mapping
        )
        
        # Convert the data to a DataFrame
        pred_df = data.get_data_as_Data_frame()
        print(pred_df)  # Debugging output
        print("Before Prediction")
        
        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Make predictions
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # Render the home page with the prediction results
        return render_template('home.html', results=results[0])

# Run the application
if __name__ == "__main__":  # Corrected variable name
    app.run(host="0.0.0.0")  # Run the app on all available interfaces

