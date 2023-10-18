import os
import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import string

app = Flask(__name__)

# Directory path for saving/loading the model and vectorizer
MODEL_DIR = 'pretrained_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'password_strength_model.joblib')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize variables for the model and vectorizer
tdif = None
loaded_model = None

# Tokenize function for TfidfVectorizer
def word(password):
    character = []
    for i in password:
        character.append(i)
    return character

# Load the dataset and create the TF-IDF vectorizer
def load_data_and_tfidf():
    data = pd.read_csv("password_strength.csv", on_bad_lines='skip')
    data = data.dropna()
    data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})
    x = np.array(data["password"])
    tdif = TfidfVectorizer(tokenizer=word)
    x = tdif.fit_transform(x)
    return tdif, x, data

# Train the model using the provided TF-IDF vectorized data
def train_model(x, y):
    model = RandomForestClassifier()
    loaded_model = model.fit(x, y)
    return loaded_model

# Load the dataset, train the model, and save the TF-IDF vectorizer
def load_data_and_train_model():
    tdif, x, data = load_data_and_tfidf()
    y = np.array(data["strength"])
    loaded_model = train_model(x, y)
    # Save the TF-IDF vectorizer
    joblib.dump(tdif, TFIDF_PATH)
    return tdif, loaded_model

@app.before_first_request
def startup():
    global tdif, loaded_model
    # Try to load the pretrained model and vectorizer if they exist
    try:
        loaded_model = joblib.load(MODEL_PATH)
        tdif = joblib.load(TFIDF_PATH)
    except FileNotFoundError:
        # If the pretrained model or vectorizer doesn't exist, train and save them
        tdif, loaded_model = load_data_and_train_model()
        joblib.dump(loaded_model, MODEL_PATH)

# Function to predict password strength
def predict_password_strength(user_input):
    # Transform user input using the TF-IDF vectorizer
    data = tdif.transform([user_input]).toarray()

    # Make predictions using the trained model
    output = loaded_model.predict(data)
    return output[0]

@app.route('/')
def index():
    return render_template('index.html')

def count_and_list_characters(password):
    characters = {
        'digits': [i+1 for i, char in enumerate(password) if char.isdigit()],
        'characters': [i+1 for i, char in enumerate(password) if char.isalpha()],
        'special_characters': [i+1 for i, char in enumerate(password) if not char.isalnum() and not char.isspace()],
        'spaces': [i+1 for i, char in enumerate(password) if char.isspace()],
    }
    counts = {key: len(value) for key, value in characters.items()}
    return counts, characters




def suggest_password_improvements(password):
    suggestions = []

    # Suggest increasing password length if it's too short
    if len(password) < 8:
        suggestions.append("Consider increasing the password length for better security.")

    # Suggest adding uppercase characters if not present
    if not any(char.isupper() for char in password):
        suggestions.append("Consider adding uppercase characters.")

    # Suggest adding lowercase characters if not present
    if not any(char.islower() for char in password):
        suggestions.append("Consider adding lowercase characters.")

    # Suggest adding special characters if not present
    if not any(char in string.punctuation for char in password):
        suggestions.append("Consider adding special characters.")

    # Suggest adding digits if not present
    if not any(char.isdigit() for char in password):
        suggestions.append("Consider adding digits.")

    if not suggestions:
        suggestions.append("No suggestions for improvement required.")

    return suggestions


# Function to predict password strength and provide suggestions
def predict_password_strength_with_suggestions(password):
    # Transform user input using the TF-IDF vectorizer
    data = tdif.transform([password]).toarray()

    # Make predictions using the trained model
    output = loaded_model.predict(data)
    
    # Get suggestions for improving the password
    suggestions = suggest_password_improvements(password)

    return output[0], suggestions

def describe_password_properties(counts):
    properties = []
    if 'digits' in counts and counts['digits']:
        properties.append(f"{counts['digits']} digits")
    if 'characters' in counts and counts['characters']:
        properties.append(f"{counts['characters']} characters")
    if 'spaces' in counts and counts['spaces']:
        properties.append(f"{counts['spaces']} spaces")
        
    if 'special_characters' in counts and counts['special_characters']:
        properties.append(f"{counts['special_characters']} special characters")
    
    return "It has "+", ".join(properties)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'user_input' in data:
        user_input = data['user_input']
        result, suggestions = predict_password_strength_with_suggestions(user_input)
        counts, characters = count_and_list_characters(user_input)
        print(counts)
        properties = describe_password_properties(counts)  # New function
        print(properties)
        response = {
            'result': result,
            'counts': counts,
            'characters_positions': characters,
            'suggestions': suggestions,
            'properties': properties 
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid JSON input'})

