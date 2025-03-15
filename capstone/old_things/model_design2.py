import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. CSV 파일 불러오기
data = pd.read_csv('./res/ideal_synthetic_fermentation_data_v4.csv')

# 데이터 타입 변환
data['temperature'] = data['temperature'].astype(float)
data['refresh_count'] = data['refresh_count'].astype(int)
data['elapsed_time'] = data['elapsed_time'].astype(int)
data['growth_percentage'] = data['growth_percentage'].astype(int)
data['remaining_time'] = data['remaining_time'].astype(int)

# 2. 원-핫 인코딩 적용
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(data[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
data = pd.concat([data, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# 3. 주요 변수에 집중하도록 특성 설정
X = data.drop(['remaining_time'], axis=1)
y = data['remaining_time']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. RandomForest 모델 학습 및 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
    'max_features': ['auto', 'sqrt']
}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled, y)

# 최적 모델 선택
best_model = grid_search.best_estimator_
print("Best model found:", best_model)

# 모델 평가
y_pred = best_model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)

# 모델 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_file = f"random_forest_model_{timestamp}.pkl"
joblib.dump((best_model, scaler, X.columns), model_file)
print(f"Model saved as '{model_file}'")
