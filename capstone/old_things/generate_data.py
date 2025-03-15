
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_ideal_synthetic_data_v4(num_samples=5000, random_state=42):
    np.random.seed(random_state)

    # 설정 가능한 온도와 혼합 비율
    temperatures = [24.0, 25.0, 26.0, 28.0]
    ratios = ['1:1:1', '1:2:2', '1:5:5', '1:10:10']
    refresh_counts = [1, 2, 3, 4, 5, 6, 7, 8]

    # 기본 시간과 감소율 설정 (지수 함수 기반)
    base_time = 2000
    decay_rate = 0.4  # 갱신 횟수가 증가할수록 경과 시간이 감소하는 비율
    saturation_point = 7  # 갱신 횟수 7 이상에서 수렴

    # 온도와 혼합 비율에 따른 가중치 설정
    temp_adjustments = {24.0: 1.3, 25.0: 1.1, 26.0: 0.9, 28.0: 0.7}
    ratio_adjustments = {'1:1:1': 0.7, '1:2:2': 0.8, '1:5:5': 0.9, '1:10:10': 1.0}

    data = []

    for _ in range(num_samples):
        # 무작위로 온도와 혼합 비율 선택
        temp = float(np.random.choice(temperatures))
        ratio = np.random.choice(ratios)
        refresh = np.random.choice(refresh_counts)

        # 갱신 횟수에 따라 지수적으로 감소하는 경과 시간 생성
        if refresh < saturation_point:
            elapsed_time = base_time * np.exp(-decay_rate * (refresh - 1))
        else:
            # 갱신 횟수가 saturation_point 이상일 경우 수렴
            elapsed_time = base_time * np.exp(-decay_rate * (saturation_point - 1))
            elapsed_time *= 0.9  # 수렴 구간에서 약간의 조정

        # 온도와 혼합 비율에 따른 가중치 적용
        elapsed_time *= temp_adjustments[temp]
        elapsed_time *= ratio_adjustments[ratio]
        elapsed_time = max(elapsed_time, 60)  # 최소 시간 제한

        # 성장 비율 생성 및 조정
        growth_percentage = np.clip(int(np.random.normal(50, 20)), 0, 100)
        if temp == 28.0:
            growth_percentage += 10
        elif temp == 24.0:
            growth_percentage -= 10

        # 남은 시간 계산
        remaining_time = int(elapsed_time * (1 - growth_percentage / 100))
        if growth_percentage == 100:
            remaining_time = 0

        data.append([temp, ratio, refresh, elapsed_time, growth_percentage, remaining_time])

    # DataFrame으로 변환
    df = pd.DataFrame(data, columns=['temperature', 'mixing_ratio', 'refresh_count', 'elapsed_time', 'growth_percentage', 'remaining_time'])
    return df

if __name__ == "__main__":
    # 데이터 생성 및 저장
    ideal_data_v4 = generate_ideal_synthetic_data_v4(num_samples=5000)
    ideal_data_v4.to_csv('./res/ideal_synthetic_fermentation_data_v4.csv', index=False)

    # 갱신 횟수와 경과 시간의 평균 관계 시각화
    avg_elapsed_time_v4 = ideal_data_v4.groupby('refresh_count')['elapsed_time'].mean()

    # # 그래프 그리기
    # plt.figure(figsize=(10, 6))
    # plt.plot(avg_elapsed_time_v4.index, avg_elapsed_time_v4.values, marker='o', color='green')
    # plt.title("Elapsed Time vs. Refresh Count (Adjusted with Saturation)")
    # plt.xlabel("Refresh Count")
    # plt.ylabel("Average Elapsed Time")
    # plt.grid(True)
    # plt.show()
