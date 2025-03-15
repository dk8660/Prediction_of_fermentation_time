import matplotlib
matplotlib.use('Agg')  # Tkinter를 사용하지 않는 백엔드로 설정

import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# 그래프 생성 함수
def create_feature_importance_plot(mixing_ratio):
    # 모델 파일 경로
    model_initial_path = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
    model_update_path = f'./model/model_update_{mixing_ratio.replace(":", "")}.joblib'

    # 모델 로드
    model_initial = joblib.load(model_initial_path)
    model_update = joblib.load(model_update_path)

    # 가중치 가져오기
    initial_features = ['Temperature', 'Refresh Count']
    update_features = ['Initial Prediction', 'Elapsed Time', 'Growth Percentage']
    all_features = initial_features + update_features
    all_weights = np.concatenate([model_initial.feature_importances_, model_update.feature_importances_])

    # 그래프 생성
    x = np.arange(len(all_features))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, all_weights, width, color='skyblue')

    # 그래프 레이블
    ax.set_xlabel('Variables')
    ax.set_ylabel('Feature Importance')
    ax.set_title(f'Feature Importance for Mixing Ratio {mixing_ratio}')
    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha='right')

    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()

    # 그래프 저장
    output_file = f'feature_importance_{mixing_ratio.replace(":", "_")}.png'
    plt.savefig(output_file)
    plt.close(fig)  # 메모리 절약을 위해 그래프 닫기
    print(f"그래프 저장 완료: {output_file}")

# 비율별 그래프 생성
for ratio in ['1:1:1', '1:2:2', '1:5:5', '1:10:10']:
    create_feature_importance_plot(ratio)
