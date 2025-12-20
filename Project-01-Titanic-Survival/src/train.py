#used to save the trained model to a file 
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from preprocess import preprocess_data

import pandas as pd

def train_models(data_path):
    df = pd.read_csv(data_path)

    X, y = preprocess_data(df)

    #train test split 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state=42
    )

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results={}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = acc
        print(f"{name}  Accuracy: {acc:.4f}")

    #save best model (Random Forest)
    best_model = models["Logistic Regression"]
    with open("../models/titanic_model.pkl","wb") as f:
        pickle.dump(best_model, f)
    
    print("best model saved as titanic_model.pkl")

if __name__=="__main__":
    train_models("../data/train.csv")

#Even though Random Forest is more powerful, Logistic Regression performed better here due to the small dataset size and strong linear relationships (gender, class). Therefore, simpler models can sometimes generalize better.    




