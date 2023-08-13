import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], target_feature_train_arr
            X_test, y_test = test_array[:, :-1], target_feature_test_arr

            models = {
                "Random Forest": RandomForestClassifier(),
                "LogisticRegression":LogisticRegression(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier(),
                #"SVC":SVC()

            }
            model_report = evaluate_models(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        models=models,
                    )
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)