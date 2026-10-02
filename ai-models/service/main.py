import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.joblib")).expanduser()
SCHEMA_PATH = Path(os.getenv("MODEL_SCHEMA_PATH", BASE_DIR / "models" / "schema.json")).expanduser()
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", BASE_DIR / "models" / "metadata.json")).expanduser()
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
logger = logging.getLogger("ai_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def input_feature_specs(feature_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    input_specs = feature_schema.get("input_features")
    if isinstance(input_specs, dict):
        return input_specs
    specs = feature_schema.get("feature_schema", {})
    required_features = feature_schema.get("required_raw_features", [])
    return {
        name: spec
        for name, spec in specs.items()
        if not required_features or name in required_features
    }


def validate_features(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must contain a JSON object under 'features'.")

    specs = input_feature_specs(schema)
    normalized: dict[str, Any] = {}
    for name, spec in specs.items():
        nullable = bool(spec.get("nullable", False))
        if name not in payload:
            if not nullable:
                raise ValueError(f"Missing required field: {name}.")
            normalized[name] = None
            continue
        value = payload[name]
        if value is None or value == "":
            if nullable:
                normalized[name] = None
                continue
            raise ValueError(f"Field '{name}' is required.")

        if "range" in spec or spec.get("type") in {"number", "integer"}:
            if isinstance(value, bool):
                raise ValueError(f"Field '{name}' must be numeric.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field '{name}' must be numeric.") from exc
            if not math.isfinite(number):
                raise ValueError(f"Field '{name}' must be finite.")
            limits = spec.get("range")
            minimum = spec.get("min", limits[0] if limits else None)
            maximum = spec.get("max", limits[1] if limits else None)
            if minimum is not None and number < float(minimum):
                raise ValueError(f"Field '{name}' is below the allowed minimum ({minimum}).")
            if maximum is not None and number > float(maximum):
                raise ValueError(f"Field '{name}' exceeds the allowed maximum ({maximum}).")
            normalized[name] = number
        else:
            text = str(value).strip()
            allowed = spec.get("allowed_values")
            if allowed and text not in allowed:
                raise ValueError(f"Field '{name}' must be one of {allowed}.")
            normalized[name] = text

    required = schema.get("required_raw_features")
    if required is None:
        required = [name for name, spec in specs.items() if not spec.get("nullable", False)]
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError(f"Missing required fields: {missing}.")
    raw_features = schema.get("required_raw_features", list(specs))
    return {name: normalized[name] for name in raw_features if name in normalized}


def parse_cors_origins() -> list[str]:
    raw_value = os.getenv("CORS_ORIGINS")
    if not raw_value:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


schema = load_json(SCHEMA_PATH)
metadata = load_json(METADATA_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="California Housing AI Service", version="2.0.0")
cors_origins = parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    logger.info("ai-service req=%s start method=%s path=%s", request_id, request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "ai-service req=%s status=%s duration_ms=%.2f",
        request_id,
        response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-service",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    target = metadata.get("target_column", schema.get("target_column", "median_house_value"))
    return {
        "model_name": metadata.get("model_name", "California Housing Price Regressor"),
        "model_version": metadata.get("model_version", "unknown"),
        "target_column": target,
        "metrics": metadata.get("metrics", {}),
        "environment": metadata.get("environment", metadata.get("environment_versions", {})),
        "model_path": str(MODEL_PATH),
        "feature_count": len(input_feature_specs(schema)),
        "schema_path": str(SCHEMA_PATH),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = request.state.request_id
    try:
        features = validate_features(payload.get("features"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_input", "detail": str(exc), "request_id": request_id},
        ) from exc

    try:
        prediction = float(model.predict(pd.DataFrame([features]))[0])
        if not math.isfinite(prediction):
            raise RuntimeError("Model returned a non-finite prediction.")
        return {
            "prediction": prediction,
            "model_version": metadata.get("model_version", "unknown"),
            "request_id": request_id,
            "target_column": metadata.get(
                "target_column", schema.get("target_column", "median_house_value")
            ),
        }
    except Exception as exc:
        logger.exception("Prediction failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "detail": "Model prediction failed.",
                "request_id": request_id,
            },
        ) from exc