import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# 백엔드를 'Agg'로 설정해 그래픽 오류 방지
matplotlib.use('Agg')

# 모델 파일 로드
model_file = './random_forest_model_20241113_203827.pkl'
loaded_model, scaler, feature_names = joblib.load(model_file)

# 예측에 사용할 데이터 준비
data = """
19,1:1:1,10,300,100
19,1:2:2,12,330,100
19,1:2:2,12,360,100
19,1:1:1,10,360,100
19,1:10:10,14,620,100
19,1:5:5,12,480,100
19,1:5:5,12,540,100
19,1:10:10,14,660,100
19,1:1:1,10,360,100
19,1:2:2,12,360,100
"""
rows = [row.split(",") for row in data.strip().split("\n")]
df = pd.DataFrame(rows, columns=['temperature', 'mixing_ratio', 'refresh_count', 'elapsed_time', 'growth_percentage'])
df['temperature'] = df['temperature'].astype(float)
df['refresh_count'] = df['refresh_count'].astype(int)
df['elapsed_time'] = df['elapsed_time'].astype(int)
df['growth_percentage'] = df['growth_percentage'].astype(int)

# One-hot 인코딩
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(df[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
df = pd.concat([df, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# 실제 경과 시간을 저장하고 예측에 사용되지 않도록 제거
actual_elapsed_times = df['elapsed_time']
df.drop(['elapsed_time'], axis=1, inplace=True)

# 'growth_percentage'를 예측을 위해 0으로 설정
df['growth_percentage'] = 0

# 누락된 열을 0으로 채움
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

df = df[feature_names]

# 데이터 스케일링
df_scaled = scaler.transform(df)

# 예측
predictions = loaded_model.predict(df_scaled)

# 결과 출력
results = pd.DataFrame({
    'temperature': df['temperature'],
    'refresh_count': df['refresh_count'],
    'predicted_elapsed_time': predictions,
    'actual_elapsed_time': actual_elapsed_times
})

# MAE 계산
mae = mean_absolute_error(results['actual_elapsed_time'], results['predicted_elapsed_time'])
print(f"Mean Absolute Error: {mae}")
print("Prediction Results:")
print(results)

# 특성 중요도 시각화
feature_importances = loaded_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in the Random Forest Model")
plt.grid(True)
plt.savefig("feature_importances_updated.png")
print("Feature importance graph saved as 'feature_importances_updated.png'")
