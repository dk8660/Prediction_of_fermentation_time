from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

# 모델 1:
# 온도와 비율, 리프레시 횟수만을 이용하여 발효종이 2배가 되는 시점만을 예측하는 모델

def train_model_initial(data, mixing_ratio):
    # growth_percentage가 0인 데이터만 필터링
    ratio_data = data[(data['mixing_ratio'] == mixing_ratio) & (data['growth_percentage'] == 0)]
    
    # 학습용 데이터 생성
    X = ratio_data[['temperature', 'refresh_count']]
    y = ratio_data['remaining_time']
    
    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomForest 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 모델 저장
    os.makedirs('./model', exist_ok=True)
    model_filename = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
    joblib.dump(model, model_filename)
    print(f"Model for {mixing_ratio} saved as {model_filename}")

# 데이터 로드 및 모델 학습
data = pd.read_csv('./res/augmented_data.csv')
mixing_ratios = ['1:1:1', '1:2:2', '1:5:5', '1:10:10'] # 비율별로 모델을 따로 학습
for ratio in mixing_ratios:
    train_model_initial(data, ratio)