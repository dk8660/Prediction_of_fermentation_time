import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# 모델 및 스케일러 불러오기
model_stage1, scaler_stage1, feature_names_stage1 = joblib.load('./model/model_initial.pkl')
model_stage2, scaler_stage2, feature_names_stage2 = joblib.load('./model/model_update.pkl')

# CSV 파일에서 데이터 불러오기
df = pd.read_csv('./res/structured_data.csv')

# One-hot encoding for 'mixing_ratio'
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(df[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
df = pd.concat([df, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# temp_refresh_interaction 추가
df['temp_refresh_interaction'] = df['temperature'] * df['refresh_count']

# 누락된 열을 추가하고, 존재하지 않는 열은 0으로 채움 (Stage 1)
for col in feature_names_stage1:
    if col not in df.columns:
        df[col] = 0

### Step 1: 첫 번째 모델 예측 ###
df_stage1 = df[feature_names_stage1]
df_scaled_stage1 = scaler_stage1.transform(df_stage1)
predictions_stage1 = model_stage1.predict(df_scaled_stage1)
predictions_stage1 = predictions_stage1.round().astype(int)

# 첫 번째 모델 예측 결과를 DataFrame에 추가
df['predicted_remaining_stage1'] = predictions_stage1

# **parabolic_factor 추가** (두 번째 모델에서 사용됨)
df['parabolic_factor'] = (1 - (df['growth_percentage'] / 100)**2)

# 누락된 열을 추가하고, 존재하지 않는 열은 0으로 채움 (Stage 2)
for col in feature_names_stage2:
    if col not in df.columns:
        df[col] = 0

### Step 2: 두 번째 모델 예측 ###
df_stage2 = df[feature_names_stage2]
df_scaled_stage2 = scaler_stage2.transform(df_stage2)
predictions_stage2 = model_stage2.predict(df_scaled_stage2)
predictions_stage2 = predictions_stage2.round().astype(int)

# 예측 결과에서 growth_percentage == 100일 때 predicted_remaining_stage2를 0으로 수정
df.loc[df['growth_percentage'] == 100, 'predicted_remaining_stage2'] = 0

# 결과 DataFrame 생성
results = pd.DataFrame({
    'temperature': df['temperature'],
    'refresh_count': df['refresh_count'],
    'elapsed_time': df['elapsed_time'],
    'growth_percentage': df['growth_percentage'],
    'remaining_time': df['remaining_time'],
    'predicted_remaining_stage1': predictions_stage1,
    'predicted_remaining_stage2': predictions_stage2
})

# Mean Absolute Error (Stage 1)
mae_stage1 = mean_absolute_error(df['remaining_time'], predictions_stage1)

# Mean Absolute Error (Stage 2)
mae_stage2 = mean_absolute_error(df['remaining_time'], predictions_stage2)

# 결과 출력
print(f"Mean Absolute Error (Stage 1): {mae_stage1:.2f}")
print(f"Mean Absolute Error (Stage 2): {mae_stage2:.2f}")
print("\nPrediction Results:")
print(results)
