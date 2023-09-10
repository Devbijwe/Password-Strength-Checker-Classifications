import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import getpass


def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  


def preprocess_and_predict():
    data = pd.read_csv("password_strength.csv", error_bad_lines=False)
    
    data = data.dropna()
    data["strength"] = data["strength"].map({0: "Weak", 
                                            1: "Medium",
                                            2: "Strong"})
    
    x = np.array(data["password"])
    y = np.array(data["strength"])

    tdif = TfidfVectorizer(tokenizer=word)
    x = tdif.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
        
   
    
    # Assuming you have xtrain, ytrain, xtest, and ytest data
    model = RandomForestClassifier()
    loaded_model=model.fit(xtrain, ytrain)

    # # Save the trained model to a file
    # filename = 'random_forest_model.pkl'
    # joblib.dump(model, filename)

    # # Load the saved model from the file
    # loaded_model = joblib.load(filename)

    # Use the loaded model for predictions
    score = loaded_model.score(xtest, ytest)
   
    tdif = TfidfVectorizer()  # You need to initialize and fit_transform your TD-IDF vectorizer here

    # Get user input
    user = input("Enter Text: ")

    # Transform user input using the TF-IDF vectorizer
    data = tdif.transform([user]).toarray()

    # Make predictions using the trained model
    output = loaded_model.predict(data)
    return output


if __name__=="__main__":
    
    result=preprocess_and_predict()
    print(result)
            