import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from preprocess import DATA_PATH, get_raw_input_preprocessor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model.joblib")
MODEL_VERSION = "2.0.0"


def train_project_model(data_path=DATA_PATH):
    # 1. Đọc dữ liệu
    df = pd.read_csv(data_path)
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']

    # 2. Chia tập Train/Test đúng tỷ lệ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Khởi tạo Top-Level Pipeline (Preprocessor + Regressor)
    preprocessor = get_raw_input_preprocessor()
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # 4. Định nghĩa không gian tìm kiếm hyperparameter
    param_dist = {
        'regressor__n_estimators': [100, 150, 200],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7]
    }

    # 5. Huấn luyện bằng CV trên tập Train để tránh rò rỉ thông tin từ tập Test
    print("Đang tìm kiếm siêu tham số qua RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    # Lưu mô hình xuất sắc nhất theo CV
    best_cv_rmse = np.sqrt(-search.best_score_)
    print(f"Mô hình tốt nhất tìm thấy với Mean CV RMSE: {best_cv_rmse:.2f}")

    best_model = search.best_estimator_

    # Refit trên toàn bộ tập Train (đã được tự động thực hiện bởi search.best_estimator_)
    # 6. Đóng gói mô hình chính thức ra file model.joblib
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    print(f"✔ Đã lưu mô hình Top-Level Pipeline thành công tại {MODEL_PATH}")

    return best_model, X_train, X_test, y_train, y_test, best_cv_rmse, search.best_params_

if __name__ == '__main__':
    train_project_model()