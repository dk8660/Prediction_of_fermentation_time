import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 데이터 불러오기
data = pd.read_csv('./res/ideal_synthetic_fermentation_data_v4.csv')

# 데이터 전처리
data['temperature'] = data['temperature'].astype(float)
data['refresh_count'] = data['refresh_count'].astype(int)

# One-hot encoding for 'mixing_ratio'
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(data[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
data = pd.concat([data, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# 상호작용 특성 추가
data['temp_refresh_interaction'] = data['temperature'] * data['refresh_count']

# 입력 변수(X)와 출력 변수(y) 설정
feature_columns = ['temperature', 'refresh_count', 'temp_refresh_interaction'] + list(ratio_encoded_df.columns)
X = data[feature_columns]
y = data['remaining_time']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# 모델과 스케일러 저장
print(f"Saving model with scikit-learn version: {sklearn.__version__}")
joblib.dump((model, scaler, feature_columns), './model/model_initial.pkl', protocol=4)
print("Initial model trained and saved.")
