import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# 백엔드를 'Agg'로 변경하여 Tkinter 문제 우회
matplotlib.use('Agg')

# 모델 및 스케일러, feature_names 불러오기
model_file = './random_forest_model_weighted_20241113_202524.pkl'
loaded_model, scaler, feature_names = joblib.load(model_file)

# 제공된 데이터를 DataFrame으로 생성
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
19,1:5:5,12,510,100
19,1:10:10,14,690,100
"""

# 데이터 정리 및 DataFrame 생성
rows = [row.split(",") for row in data.strip().split("\n")]
df = pd.DataFrame(rows, columns=['temperature', 'mixing_ratio', 'refresh_count', 'elapsed_time', 'growth_percentage'])
df['temperature'] = df['temperature'].astype(float)
df['refresh_count'] = df['refresh_count'].astype(int)
df['elapsed_time'] = df['elapsed_time'].astype(int)
df['growth_percentage'] = df['growth_percentage'].astype(int)

# 원-핫 인코딩 적용
encoder = OneHotEncoder(sparse_output=False)
ratio_encoded = encoder.fit_transform(df[['mixing_ratio']])
ratio_encoded_df = pd.DataFrame(ratio_encoded, columns=encoder.get_feature_names_out(['mixing_ratio']))
df = pd.concat([df, ratio_encoded_df], axis=1).drop(['mixing_ratio'], axis=1)

# 실제 elapsed_time을 따로 저장하고, 예측을 위해 제거
actual_elapsed_times = df['elapsed_time'].copy()
df.drop(['elapsed_time'], axis=1, inplace=True)

# 예측을 위해 'growth_percentage'를 0으로 설정
df['growth_percentage'] = 0

# 상호작용 특성 추가
df['temp_refresh_interaction'] = df['temperature'] * df['refresh_count']

# 누락된 열을 0으로 채워 넣기
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

# 열 순서를 학습된 모델의 feature_names와 맞추기
df = df[feature_names]

# 스케일링 적용
df_scaled = scaler.transform(df)

# 예측 수행
predictions = loaded_model.predict(df_scaled)

# 음수 예측 값은 0으로 조정
predictions = [max(0, p) for p in predictions]

# 결과 비교
results = pd.DataFrame({
    'temperature': df['temperature'],
    'refresh_count': df['refresh_count'],
    'predicted_elapsed_time': predictions,
    'actual_elapsed_time': actual_elapsed_times
})

# MAE 계산 및 출력
mae = mean_absolute_error(results['actual_elapsed_time'], results['predicted_elapsed_time'])
print(f"Mean Absolute Error: {mae}")

# 결과 출력
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
