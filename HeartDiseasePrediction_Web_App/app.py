from flask import Flask, render_template, request, jsonify
import pickle


app = Flask(__name__)

# Load the saved pickle model
model = pickle.load(open('heart_disease_model_ickle.pkl', 'rb'))

# Define a route to handle the index page
@app.route('/')
def index():
    return render_template('index.html')
# Define a route to handle the form submission and predict accuracy
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    print(int(request.json['age']))
    age = int(request.json['age'])
    sex = int(request.json['sex'])
    cp = int(request.json['cp'])
    trestbps = int(request.json['trestbps'])
    chol = int(request.json['chol'])
    fbs = int(request.json['fbs'])
    restecg = int(request.json['restecg'])
    thalach = int(request.json['thalach'])
    exang = int(request.json['exang'])
    oldpeak = float(request.json['oldpeak'])
    slope = int(request.json['slope'])
    ca = int(request.json['ca'])
    thal = int(request.json['thal'])
    # Get other form inputs in a similar way

    # Create a feature vector using the form inputs
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    # Add other features to the list
    
    # Make predictions using the model
    accuracy = model.predict([features])[0]
    print(accuracy)
    # Return the accuracy result as JSON response
    return str(accuracy)

if __name__ == '__main__':
    app.run(debug=True)
