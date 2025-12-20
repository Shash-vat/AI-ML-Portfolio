import pandas as pd
import numpy as np

def preprocess_data(df, is_train=True):
    #drop columns with too many missing values or no predictive power 
    df = df.drop(columns=["Cabin","Ticket","Name"],errors="ignore")

    #Fill missing Age wtih median 
    df["Age"]=df["Age"].fillna(df["Age"].median())

    df["Fare"]=df["Fare"].fillna(df["Fare"].median())

    #Fill missing embarked with the mode
    df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

    #Encode sex(binary encoding)
    df["Sex"]=df["Sex"].map({"male":0, "female":1})

    #One-Hot encode embarked (drop_first to avoid dummy variable trap)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    #Dropping passenger id for training data but keeping it for test data
    if "PassengerId" in df.columns:
        passenger_id = df["PassengerId"]
        df = df.drop(columns=["PassengerId"])
    else :
        passenger_id = None

    #returning data as per whether it is training or test data 
    if is_train:
        X = df.drop(columns=["Survived"]);
        y = df["Survived"]
        return X, y
    else:
        return df, passenger_id




