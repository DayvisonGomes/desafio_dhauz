"""Download dataset utilities (Kaggle)"""
import os
import pandas as pd
try:
    import kagglehub
except Exception:
    kagglehub = None

from ..config import DATA_DIR


class DataDownloader:
    KAGGLE_DATASET = "adisongoh/it-service-ticket-classification-dataset"
    CSV_FILE = "all_tickets_processed_improved_v3.csv"

    @staticmethod
    def download(output_dir: str = DATA_DIR) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)
        if kagglehub is None:
            raise RuntimeError("kagglehub not available; install dependencies or provide CSV manually")
        path = kagglehub.dataset_download(DataDownloader.KAGGLE_DATASET)
        df = pd.read_csv(os.path.join(path, DataDownloader.CSV_FILE))
        df = df.rename(columns={"Document": "text", "Topic_group": "class"})
        return df
