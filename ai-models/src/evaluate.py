import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import sklearn
from preprocess import DATA_PATH, generate_schema, load_dataset
from train import MODEL_VERSION

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

def evaluate_and_generate_artifacts(best_model, X_train, X_test, y_train, y_test, cv_rmse, best_params):
    # 1. Đánh giá một lần duy nhất trên tập kiểm thử chưa từng thấy
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print("--- ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP HOÀN TOÀN ĐỘC LẬP (TEST SET) ---")
    print(f"Test RMSE: {test_rmse:.2f} USD")
    print(f"Test MAE: {test_mae:.2f} USD")
    print(f"Test R2 Score: {test_r2:.4f}")

    # 2. Tạo schema.json từ cùng generator được dùng trong các kiểm thử.
    schema = generate_schema(load_dataset(DATA_PATH))
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'schema.json'), 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=4, ensure_ascii=False)
    print("✔ Đã tạo tệp models/schema.json")

    # 3. Tạo metadata.json
    metadata = {
        "model_name": "California Housing Price Regressor",
        "model_version": MODEL_VERSION,
        "target_column": "median_house_value",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_hyperparameters": {k.replace('regressor__', ''): v for k, v in best_params.items()},
        "metrics": {
            "train_cv_rmse": float(cv_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2)
        },
        "environment": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__
        }
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("✔ Đã tạo tệp models/metadata.json")

if __name__ == '__main__':
    # Chạy quy trình liên hoàn
    from train import train_project_model
    best_model, X_train, X_test, y_train, y_test, cv_rmse, best_params = train_project_model(DATA_PATH)
    evaluate_and_generate_artifacts(best_model, X_train, X_test, y_train, y_test, cv_rmse, best_params)