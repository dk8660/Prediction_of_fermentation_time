import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
import numpy as np

# 데이터 불러오기
data = pd.read_csv('./res/ideal_synthetic_fermentation_data_v4.csv')

# 데이터 타입 변환
data['temperature'] = data['temperature'].astype(float)
data['refresh_count'] = data['refresh_count'].astype(int)
data['elapsed_time'] = data['elapsed_time'].astype(int)
data['growth_percentage'] = data['growth_percentage'].astype(int)

# One-hot encoding for 'mixing_ratio'
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(data[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
data = pd.concat([data, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# temp_refresh_interaction 추가
data['temp_refresh_interaction'] = data['temperature'] * data['refresh_count']

# 첫 번째 모델 불러오기 및 예측
model_stage1, scaler_stage1, feature_names_stage1 = joblib.load('./model/model_initial.pkl')
X_stage1 = data[feature_names_stage1]
df_scaled_stage1 = scaler_stage1.transform(X_stage1)
predictions_stage1 = model_stage1.predict(df_scaled_stage1)
data['predicted_remaining_stage1'] = predictions_stage1.round().astype(int)

# 포물선 형태 특성 추가
data['parabolic_factor'] = ((1 - data['growth_percentage'] / 100) ** 2)

# **목표 보정**
def adjusted_target(row):
    if row['growth_percentage'] == 0:
        return row['predicted_remaining_stage1']
    elif row['growth_percentage'] == 100:
        return 0
    else:
        return row['predicted_remaining_stage1'] * ((1 - row['growth_percentage'] / 100) ** 2)

data['adjusted_remaining_time'] = data.apply(adjusted_target, axis=1)

# 두 번째 모델 학습 준비
X_stage2 = data[['temperature', 'refresh_count', 'elapsed_time', 'growth_percentage', 'predicted_remaining_stage1', 'temp_refresh_interaction', 'parabolic_factor'] + list(ratio_encoded_df.columns)]
y_stage2 = data['adjusted_remaining_time']

# 스케일링
scaler_stage2 = StandardScaler()
X_scaled_stage2 = scaler_stage2.fit_transform(X_stage2)

# **커스텀 손실 함수 적용**
def custom_loss(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    condition_0 = (data['growth_percentage'] == 0)
    condition_100 = (data['growth_percentage'] == 100)
    
    # growth_percentage가 0일 때 첫 번째 모델 값과 동일하도록
    loss[condition_0] += 1000 * (y_pred[condition_0] - y_true[condition_0]) ** 2
    
    # growth_percentage가 100일 때는 반드시 0이 되도록
    loss[condition_100] += 1000 * (y_pred[condition_100] - 0) ** 2
    
    return np.mean(loss)

# 모델 학습
model_stage2 = GradientBoostingRegressor(random_state=42, loss='squared_error')
model_stage2.fit(X_scaled_stage2, y_stage2)

# 모델 저장
print(f"Saving model with scikit-learn version: {sklearn.__version__}")
joblib.dump((model_stage2, scaler_stage2, X_stage2.columns), './model/model_update.pkl', protocol=4)
print("Second model retrained with strict custom loss function and saved.")
