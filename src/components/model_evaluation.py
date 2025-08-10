# src/components/model_evaluation.py
import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.utils import load_object
from src.logger.logger import logging
from src.exception.exception import customexception


@dataclass
class EvalConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    test_array_path: str = os.path.join("artifacts", "test_array.npy")


class ModelEvaluation:
    def __init__(self):
        self.cfg = EvalConfig()
        logging.info("Model evaluation started")

    @staticmethod
    def _metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, mae, r2

    def initiate_model_evaluation(self):
        try:
            # Load artifacts
            test_array = np.load(self.cfg.test_array_path)
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            model = load_object(self.cfg.model_path)
            model_name = type(model).__name__

            # Predict & metrics
            y_pred = model.predict(X_test)
            rmse, mae, r2 = self._metrics(y_test, y_pred)

            # ---- Console report ----
            print("\n" + "=" * 66)
            print("MODEL EVALUATION REPORT".center(66))
            print("=" * 66)
            print(f"\nModel Loaded From : {self.cfg.model_path}")
            print(f"Detected Model    : {model_name}")
            print(f"Test Array Source : {self.cfg.test_array_path}\n")
            print("Metrics (on held-out test set):\n")
            print(f"  • R²   : {r2:.5f}")
            print(f"  • RMSE : {rmse:.2f}")
            print(f"  • MAE  : {mae:.2f}")
            print("=" * 66 + "\n")
            # -----------------------

        except Exception as e:
            raise customexception(e, sys)


if __name__ == "__main__":
    ModelEvaluation().initiate_model_evaluation()
    print("✅ Evaluation complete.")
