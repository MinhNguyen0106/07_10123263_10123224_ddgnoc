import json
import pandas as pd
import pytest
import sys
from pathlib import Path

AI_MODELS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_MODELS_DIR / "src"))

from preprocess import DATA_PATH, REQUIRED_RAW_FEATURES, generate_schema, load_dataset


def test_generated_schema_matches_raw_model_input_contract():
    data = {
        "longitude": [-122.0, -121.0],
        "latitude": [37.0, 38.0],
        "housing_median_age": [10.0, 20.0],
        "total_rooms": [100.0, 200.0],
        "total_bedrooms": [None, 50.0],
        "population": [300.0, 400.0],
        "households": [100.0, 150.0],
        "median_income": [2.0, 4.0],
        "ocean_proximity": ["INLAND", "NEAR BAY"],
        "median_house_value": [100000.0, 200000.0],
    }

    schema = generate_schema(pd.DataFrame(data))

    assert list(schema["input_features"]) == REQUIRED_RAW_FEATURES
    assert list(schema) == ["target_column", "input_features"]
    assert "rooms_per_household" not in schema["input_features"]
    assert schema["input_features"]["total_bedrooms"]["nullable"] is True
    assert schema["input_features"]["total_bedrooms"]["range"] == [50.0, 50.0]
    assert schema["input_features"]["ocean_proximity"]["allowed_values"] == [
        "INLAND",
        "NEAR BAY",
    ]


def test_schema_generation_reports_missing_raw_features():
    with pytest.raises(ValueError, match="missing features"):
        generate_schema(pd.DataFrame({"longitude": [0.0]}))


def test_checked_in_schema_matches_generator():
    artifact_path = AI_MODELS_DIR / "models" / "schema.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert generate_schema(load_dataset(DATA_PATH)) == artifact
