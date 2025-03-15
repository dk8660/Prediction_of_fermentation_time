from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os
import numpy as np

# 모델 2:
# 모델 1의 결과를 바탕으로 경과 시간과 현재 성장 정도를 고려하여 2배가 될 때까지 앞으로 시간이 얼마나 남았는지 실시간으로 예측하는 모델
# 현재 성장이 100%가 아닐 때 남은 시간이 0분이거나, 성장이 100%인데도 남은 시간이 0%가 아닌 문제가 있음
# 이는 학습 데이터의 부족도 원인
# 해결을 위해 강제로 성장이 100%가 아닐 경우 남은 시간을 1이상으로 설정하고, 100%인 경우 0으로 설정
# 강제로 예측 데이터를 수정한 것이라 근본적인 해결책은 아니긴 하나, 추후에 많아진 데이터를 기반으로 학습 했을 때 바꾸면 됨
# 또한 서비스 품질을 유지하기 위해 (오류 또는 사용자 혼란 방지를 위해) 이 해결책은 향상된 모델에도 적용하는 것이 좋을 수 있음

def train_model_update(data, mixing_ratio):
    # 데이터 필터링
    ratio_data = data[data['mixing_ratio'] == mixing_ratio]
    
    # model_initial 로드
    model_initial_filename = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
    model_initial = joblib.load(model_initial_filename)
    
    # model_initial 예측값 추가
    ratio_data = ratio_data.copy()
    ratio_data['initial_prediction'] = model_initial.predict(ratio_data[['temperature', 'refresh_count']])
    
    # `adjusted_remaining_time` 설정 (growth_percentage가 100일 경우 0으로 설정)
    ratio_data['adjusted_remaining_time'] = ratio_data.apply(
        lambda row: 0 if row['growth_percentage'] == 100 else row['remaining_time'], axis=1
    )
    
    # 가중치 설정: `growth_percentage`가 100일 경우 더 큰 가중치 부여
    def calculate_weight(gp):
        return 10 if gp == 100 else 1

    ratio_data['weights'] = ratio_data['growth_percentage'].apply(calculate_weight)

    # 입력(X)과 출력(y) 설정
    X = ratio_data[['initial_prediction', 'elapsed_time', 'growth_percentage']]
    y = ratio_data['adjusted_remaining_time']
    weights = ratio_data['weights']
    
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    # GradientBoostingRegressor 모델 학습 (가중치 적용)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights_train)

    # 모델 저장
    os.makedirs('./model', exist_ok=True)
    model_filename = f'./model/model_update_{mixing_ratio.replace(":", "")}.joblib'
    joblib.dump(model, model_filename)
    print(f"Model for {mixing_ratio} saved as {model_filename}")

# 데이터 로드 및 모델 학습
data = pd.read_csv('./res/filtered_original_data.csv')
mixing_ratios = ['1:1:1', '1:2:2', '1:5:5', '1:10:10']
for ratio in mixing_ratios:
    train_model_update(data, ratio)
