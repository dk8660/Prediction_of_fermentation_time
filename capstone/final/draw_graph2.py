import matplotlib
matplotlib.use('Agg')  # Tkinter를 사용하지 않는 백엔드로 설정

import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Feature Importance 그래프 생성 함수 (개별 모델용)
def create_individual_feature_importance_plot(model_path, features, title, output_file):
    # 모델 로드
    model = joblib.load(model_path)

    # 가중치 가져오기
    weights = model.feature_importances_

    # 그래프 생성
    x = np.arange(len(features))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, weights, width, color='skyblue')

    # 그래프 레이블
    ax.set_xlabel('Variables')
    ax.set_ylabel('Feature Importance')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')

    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()

    # 그래프 저장
    plt.savefig(output_file)
    plt.close(fig)  # 메모리 절약을 위해 그래프 닫기
    print(f"그래프 저장 완료: {output_file}")

# 비율별 Feature Importance 그래프 생성
for ratio in ['1:1:1', '1:2:2', '1:5:5', '1:10:10']:
    # 모델 파일 경로
    model_initial_path = f'./model/model_initial_{ratio.replace(":", "")}.joblib'
    model_update_path = f'./model/model_update_{ratio.replace(":", "")}.joblib'

    # Feature Importance for model_initial
    initial_features = ['Temperature', 'Refresh Count']
    initial_title = f'Feature Importance for model_initial ({ratio})'
    initial_output_file = f'./graph/feature_importance_model_initial_{ratio.replace(":", "_")}.png'
    create_individual_feature_importance_plot(model_initial_path, initial_features, initial_title, initial_output_file)

    # Feature Importance for model_update
    update_features = ['Initial Prediction', 'Elapsed Time', 'Growth Percentage']
    update_title = f'Feature Importance for model_update ({ratio})'
    update_output_file = f'./graph/feature_importance_model_update_{ratio.replace(":", "_")}.png'
    create_individual_feature_importance_plot(model_update_path, update_features, update_title, update_output_file)
