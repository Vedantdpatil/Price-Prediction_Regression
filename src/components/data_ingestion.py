# src/pipeline/data_ingestion.py
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger.logger import logging
from src.exception.exception import customexception


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.cfg = DataIngestionConfig()
        self.local_dataset_path = "/Users/vedantdilip/Documents/ML_Projects/Price-Prediction_Regression/GemstoneData.csv"

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Reads local dataset, renames columns, saves raw.csv,
        and splits into train.csv / test.csv.
        Returns (train_path, test_path).
        """
        logging.info("Data ingestion started (Local Dataset)")

        try:
            # Read dataset
            logging.info(f"Reading local dataset from {self.local_dataset_path}")
            df = pd.read_csv(self.local_dataset_path)
            logging.info(f"Dataset shape: {df.shape}")

            if df.empty:
                raise ValueError("Local dataset is empty.")

            # Rename columns
            rename_map = {"x": "length", "y": "width", "z": "depth_mm"}
            df = df.rename(columns=rename_map)
            logging.info(f"Columns renamed: {rename_map}")

            # Save raw.csv
            os.makedirs(os.path.dirname(self.cfg.raw_data_path), exist_ok=True)
            df.to_csv(self.cfg.raw_data_path, index=False)
            logging.info(f"Raw dataset saved -> {self.cfg.raw_data_path}")

            # Train/Test split
            train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
            train_df.to_csv(self.cfg.train_data_path, index=False)
            test_df.to_csv(self.cfg.test_data_path, index=False)
            logging.info(f"Train/Test split done: train={train_df.shape}, test={test_df.shape}")

            logging.info("Data ingestion finished successfully")
            return self.cfg.train_data_path, self.cfg.test_data_path

        except Exception as e:
            logging.exception("Data ingestion failed")
            raise customexception(e, sys)


if __name__ == "__main__":
    ing = DataIngestion()
    train_path, test_path = ing.initiate_data_ingestion()
    print("Artifacts:", train_path, test_path)
