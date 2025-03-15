import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import class_weight
import numpy as np

# 1. CSV 파일 불러오기
data = pd.read_csv('./res/ideal_synthetic_fermentation_data_v4.csv')

# 데이터 타입 수정
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

# 3. 입력 변수(X)와 출력 변수(y) 설정
X = data.drop(['remaining_time', 'elapsed_time'], axis=1)
y = data['remaining_time']

# 4. 데이터 스케일링 추가
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 특성 가중치를 조정하여 모델 학습
model = RandomForestRegressor(
    n_estimators=300, 
    max_depth=20, 
    min_samples_split=5, 
    max_features='sqrt', 
    random_state=42
)

# 특정 특성의 가중치를 인위적으로 조정
weights = np.ones(X_scaled.shape[1])
temp_index = list(X.columns).index('temperature')
mix_index_start = list(X.columns).index('mixing_ratio_1:1:1')
mix_index_end = list(X.columns).index('mixing_ratio_1:10:10') + 1

# 온도와 혼합 비율의 중요도를 높이기 위해 가중치 조정
weights[temp_index] = 3.0
weights[mix_index_start:mix_index_end] = 2.0

model.fit(X_scaled * weights, y)

# 6. 모델 평가
y_pred = model.predict(X_scaled * weights)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)

# 7. 모델 및 스케일러, 열 이름 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"random_forest_model_weighted_{timestamp}.pkl"
joblib.dump((model, scaler, X.columns), file_name)
print(f"Model saved as '{file_name}'")
