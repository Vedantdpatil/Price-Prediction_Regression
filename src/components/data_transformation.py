# src/components/data_transformation.py
import os
import pickle
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.logger.logger import logging
from src.exception.exception import customexception
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_array_path: str = os.path.join("artifacts", "train_array.npy")
    test_array_path: str  = os.path.join("artifacts", "test_array.npy")

class DataTransformation:
    def __init__(self):
        self.cfg = DataTransformationConfig()
        self.cat_cols = ["cut", "color", "clarity"]
        self.num_cols = ["carat", "depth", "table", "length", "width", "depth_mm"]

        self.cut_cats = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        self.color_cats = ["D", "E", "F", "G", "H", "I", "J"]
        self.clarity_cats = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    def get_data_transformation(self) -> ColumnTransformer:
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[self.cut_cats, self.color_cats, self.clarity_cats])),
                ("scaler", StandardScaler()),
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, self.num_cols),
                ("cat", cat_pipeline, self.cat_cols),
            ])

            logging.info("Preprocessor built successfully.")
            return preprocessor
        except Exception as e:
            logging.exception("Failed to build preprocessor")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path: str, test_path: str):
        try:
            # ensure artifacts dir exists
            os.makedirs(os.path.dirname(self.cfg.preprocessor_obj_file_path), exist_ok=True)

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train/test CSVs.")

            # Validate schema early (helps catch typos)
            required = set(self.cat_cols + self.num_cols + ["price", "id"])
            missing_train = required.difference(train_df.columns)
            missing_test = required.difference(test_df.columns)
            if missing_train or missing_test:
                raise ValueError(f"Missing columns. "
                                 f"train missing: {sorted(missing_train)} | "
                                 f"test missing: {sorted(missing_test)}")

            preproc = self.get_data_transformation()

            target = "price"
            drop_cols = [target, "id"]

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target]
            X_test  = test_df.drop(columns=drop_cols)
            y_test  = test_df[target]

            X_train_t = preproc.fit_transform(X_train)
            X_test_t  = preproc.transform(X_test)
            logging.info("Applied preprocessing to train/test.")

            train_arr = np.c_[X_train_t, np.array(y_train)]
            test_arr  = np.c_[X_test_t,  np.array(y_test)]

            # save artifacts
            save_object(self.cfg.preprocessor_obj_file_path, preproc)
            np.save(self.cfg.train_array_path, train_arr)
            np.save(self.cfg.test_array_path,  test_arr)

            logging.info(f"Saved preprocessor to {os.path.abspath(self.cfg.preprocessor_obj_file_path)}")
            logging.info(f"Saved train array to {os.path.abspath(self.cfg.train_array_path)}")
            logging.info(f"Saved test array to  {os.path.abspath(self.cfg.test_array_path)}")

            # keep returns for backward compatibility
            return train_arr, test_arr
        except Exception as e:
            logging.exception("initialize_data_transformation failed")
            raise customexception(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    ing = DataIngestion()
    train_p, test_p = ing.initiate_data_ingestion()
    DataTransformation().initialize_data_transformation(train_p, test_p)
    print("✅ preprocessor.pkl, train_array.npy, test_array.npy generated.")
