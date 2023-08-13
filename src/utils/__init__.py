import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)  
            y_test_pred = best_model.predict(X_test)    

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                    'best_params': gs.best_params_,
                    'train_model_score': train_model_score,
                    'test_model_score': test_model_score
                }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)