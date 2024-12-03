from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# https://www.kaggle.com/code/dhirajyadav079/student-placement-score
# this is the link of model 
# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        hours = float(request.form['hours'])  # Get hours from form
        prediction = model.predict(np.array([[hours]]))  # Predict the score
        return render_template('index.html', prediction_text=f'Predicted Score: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
