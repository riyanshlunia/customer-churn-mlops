import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from src.config import DATASET_CONFIGS, DATA_DIR, ARTIFACTS_DIR


class DataPipeline:
    """Universal Feature Engineering & Preprocessing Pipeline for Real-World Churn Datasets."""

    def __init__(self, dataset_key: str = "telco"):
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset_key: {dataset_key}. Options: {list(DATASET_CONFIGS.keys())}")
        self.dataset_key = dataset_key
        self.config = DATASET_CONFIGS[dataset_key]
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: list = []

    def load_raw_data(self) -> pd.DataFrame:
        """Load real-world raw CSV dataset and handle dataset-specific data quirks."""
        file_path = DATA_DIR / self.config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file '{self.config['file_name']}' not found in '{DATA_DIR}'. "
                "Please place the dataset CSV file in the data folder."
            )

        df = pd.read_csv(file_path)

        # =========================================================================
        # REAL-WORLD DATASET QUIRKS & SANITIZATION
        # =========================================================================

        # QUIRK 1 (Telco): TotalCharges contains whitespace strings " " for 11 customers with tenure == 0.
        # This causes pandas to parse TotalCharges as 'object' dtype instead of float64.
        if "TotalCharges" in df.columns:
            # Replace whitespace strings ' ' with NaN and coerce to numeric float64
            df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # QUIRK 2 (Telco): SeniorCitizen is coded as int 0/1 rather than categorical string ("Yes"/"No").
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

        # QUIRK 3 (Bank): Clean string column whitespace if present
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fit preprocessor on real dataframe and return processed X matrix, y vector, and raw features df."""
        target_col = self.config["target_col"]

        # Parse target variable (handles both string 'Yes'/'No' and integer 0/1)
        if df[target_col].dtype == object:
            y = (df[target_col].str.strip().str.lower() == "yes").astype(int).values
        else:
            y = df[target_col].values

        num_cols = self.config["numerical_features"]
        cat_cols = self.config["categorical_features"]

        X_df = df[num_cols + cat_cols].copy()

        # Build sklearn column transformer: StandardScaler for numerical, OneHotEncoder for categorical
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols),
            ]
        )

        X_proc = self.preprocessor.fit_transform(X_df)

        # Retrieve feature names after OneHotEncoding
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        self.feature_names = num_cols + encoded_cat_names

        # Save preprocessor artifact
        joblib.dump(self.preprocessor, ARTIFACTS_DIR / f"{self.dataset_key}_preprocessor.joblib")
        joblib.dump(self.feature_names, ARTIFACTS_DIR / f"{self.dataset_key}_feature_names.joblib")

        return X_proc, y, X_df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new inference dataframe using saved preprocessor, filling missing features with defaults."""
        if self.preprocessor is None:
            proc_path = ARTIFACTS_DIR / f"{self.dataset_key}_preprocessor.joblib"
            if proc_path.exists():
                self.preprocessor = joblib.load(proc_path)
            else:
                raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        num_cols = self.config["numerical_features"]
        cat_cols = self.config["categorical_features"]

        X_df = df.copy()

        # Fill missing numerical columns with 0.0
        for col in num_cols:
            if col not in X_df.columns:
                X_df[col] = 0.0
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

        # Fill missing categorical columns with fallback default strings
        for col in cat_cols:
            if col not in X_df.columns:
                X_df[col] = "No" if col != "gender" and col != "Geography" else ("Male" if col == "gender" else "France")

        X_df = X_df[num_cols + cat_cols]
        return self.preprocessor.transform(X_df)

    def get_train_val_test_splits(
        self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """Load dataset and return stratified splits for train, validation, and baseline reference data."""
        df = self.load_raw_data()
        X_proc, y, X_raw = self.fit_transform(df)

        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test, X_train_val_raw, X_test_raw = train_test_split(
            X_proc, y, X_raw, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: Train vs Validation
        val_relative_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, X_train_raw, X_val_raw = train_test_split(
            X_train_val, y_train_val, X_train_val_raw, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
        )

        # Save training reference baseline CSV for drift monitoring
        ref_path = ARTIFACTS_DIR / f"{self.dataset_key}_reference_baseline.csv"
        X_train_raw.to_csv(ref_path, index=False)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_raw": X_train_raw,
            "X_val_raw": X_val_raw,
            "X_test_raw": X_test_raw,
        }


if __name__ == "__main__":
    pipeline = DataPipeline("telco")
    splits = pipeline.get_train_val_test_splits()
    print("Real Data Pipeline Verification (Telco Dataset):")
    print(f"  Total Raw Rows:  {len(pipeline.load_raw_data()):,}")
    print(f"  X_train shape:   {splits['X_train'].shape}")
    print(f"  X_val shape:     {splits['X_val'].shape}")
    print(f"  X_test shape:    {splits['X_test'].shape}")
    print(f"  Feature count:   {len(pipeline.feature_names)}")
