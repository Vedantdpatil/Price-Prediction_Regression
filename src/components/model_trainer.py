# src/components/model_trainer.py
import os
import sys
import numpy as np
from dataclasses import dataclass

from src.logger.logger import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting features/target")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Models aligned with your EDA
            models = {
                "XGBoost": XGBRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42),
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
            }

            # Evaluate and pick best by R²
            report = evaluate_model(X_train, y_train, X_test, y_test, models)
            best_name = max(report, key=report.get)
            best_score = report[best_name]
            best_model = models[best_name]

            print(report)
            print(f"\nBest Model: {best_name} | R2: {best_score:.5f}\n")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
        except Exception as e:
            raise customexception(e, sys)


if __name__ == "__main__":
    # Minimal, straight run: load arrays and train
    train_array = np.load("artifacts/train_array.npy")
    test_array  = np.load("artifacts/test_array.npy")

    ModelTrainer().initate_model_training(train_array, test_array)
    print("✅ Training done. Saved best model to artifacts/model.pkl")
