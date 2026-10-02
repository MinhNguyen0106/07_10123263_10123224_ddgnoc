from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "housing.csv.zip"

TARGET_COLUMN = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
REQUIRED_RAW_FEATURES = RAW_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
MODEL_FEATURE_NAMES = RAW_NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
MODEL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_FEATURES


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the ratio features expected by the serialized housing pipeline."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> HousingFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            X_copy["rooms_per_household"] = (
                X_copy["total_rooms"] / X_copy["households"].replace(0, 1)
            )
            X_copy["bedrooms_per_room"] = (
                X_copy["total_bedrooms"] / X_copy["total_rooms"].replace(0, 1)
            )
            X_copy["population_per_household"] = (
                X_copy["population"] / X_copy["households"].replace(0, 1)
            )
            return X_copy

        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != len(RAW_NUMERICAL_FEATURES):
            raise ValueError(
                "HousingFeatureEngineer expects the eight raw numerical features "
                "in RAW_NUMERICAL_FEATURES order."
            )

        households = np.where(values[:, 6] == 0, 1, values[:, 6])
        total_rooms = np.where(values[:, 3] == 0, 1, values[:, 3])
        engineered = np.column_stack(
            (
                values[:, 3] / households,
                values[:, 4] / total_rooms,
                values[:, 5] / households,
            )
        )
        return np.column_stack((values, engineered))


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the project dataset from the original zip artifact."""
    return pd.read_csv(Path(path))


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def impute_total_bedrooms(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    strategy: str = "median",
):
    """Mirror the notebook: impute total_bedrooms before engineered ratios."""
    train_copy = X_train.copy()
    test_copy = X_test.copy() if X_test is not None else None
    imputer = SimpleImputer(strategy=strategy)

    train_copy["total_bedrooms"] = imputer.fit_transform(train_copy[["total_bedrooms"]])
    if test_copy is not None:
        test_copy["total_bedrooms"] = imputer.transform(test_copy[["total_bedrooms"]])
        return train_copy, test_copy, imputer
    return train_copy, imputer


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create the same ratio features used by the original notebook."""
    X_copy = X.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        X_copy["rooms_per_household"] = X_copy["total_rooms"] / X_copy["households"]
        X_copy["bedrooms_per_room"] = X_copy["total_bedrooms"] / X_copy["total_rooms"]
        X_copy["population_per_household"] = X_copy["population"] / X_copy["households"]
    X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
    return X_copy


def prepare_feature_frames(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_imputed, X_test_imputed, imputer = impute_total_bedrooms(X_train, X_test)
    return add_engineered_features(X_train_imputed), add_engineered_features(X_test_imputed), imputer


def build_preprocessor() -> ColumnTransformer:
    """Build the fitted-time preprocessing pipeline from ML_Project.ipynb."""
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, MODEL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_raw_input_preprocessor() -> ColumnTransformer:
    """Build preprocessing for the raw 9-column inference input."""
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("engineer", HousingFeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, RAW_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_training_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_fe, X_test_fe, imputer = prepare_feature_frames(X_train, X_test)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "y_train": y_train,
        "y_test": y_test,
        "imputer": imputer,
        "preprocessor": preprocessor,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
    }


def build_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    feature_schema: dict[str, Any] = {}
    for column in REQUIRED_RAW_FEATURES:
        if column in CATEGORICAL_FEATURES:
            feature_schema[column] = {
                "type": "object",
                "nullable": bool(df[column].isna().any()),
                "allowed_values": sorted(df[column].dropna().astype(str).unique().tolist()),
            }
            continue

        values = df[column].dropna()
        feature_schema[column] = {
            "type": str(df[column].dtype),
            "nullable": bool(df[column].isna().any()),
            "range": [
                float(values.min()) if not values.empty else None,
                float(values.max()) if not values.empty else None,
            ],
        }
    return feature_schema


def generate_schema(df: pd.DataFrame) -> dict[str, Any]:
    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    missing_features = set(REQUIRED_RAW_FEATURES) - set(feature_df.columns)
    if missing_features:
        raise ValueError(f"Cannot generate schema; missing features: {sorted(missing_features)}")
    return {
        "target_column": TARGET_COLUMN,
        "input_features": build_feature_schema(feature_df),
    }
