import pandas as pd
import joblib
import numpy as np
from collections import defaultdict

# csv 파일을 이용하여 많은 데이터를 한 번에 예측하고 정확도 측정
def predict_remaining_time_with_model_accuracy(input_csv):
    data = pd.read_csv(input_csv)
    predictions = []
    absolute_errors = defaultdict(list)  # 모델별 절대 오차 저장
    correct_predictions = defaultdict(int)  # 모델별 정확한 예측 개수 저장
    total_data_count = defaultdict(int)  # 모델별 데이터 개수 저장

    for index, row in data.iterrows():
        mixing_ratio = row['mixing_ratio']
        
        # model_initial 예측
        model_initial_filename = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
        model_initial = joblib.load(model_initial_filename)
        
        initial_features = pd.DataFrame([[row['temperature'], row['refresh_count']]], columns=['temperature', 'refresh_count'])
        initial_prediction = model_initial.predict(initial_features)[0]

        # model_update 예측
        model_update_filename = f'./model/model_update_{mixing_ratio.replace(":", "")}.joblib'
        model_update = joblib.load(model_update_filename)
        
        update_features = pd.DataFrame(
            [[initial_prediction, row['elapsed_time'], row['growth_percentage']]], 
            columns=['initial_prediction', 'elapsed_time', 'growth_percentage']
        )
        update_prediction = int(round(model_update.predict(update_features)[0]))

        # 강제 조정 적용
        if row['growth_percentage'] == 100:
            update_prediction = 0  # 성장 비율이 100이면 무조건 0
        elif row['growth_percentage'] < 100 and update_prediction == 0:
            update_prediction = 1  # 성장 비율이 100이 아닌데 0이면 1로 조정

        # 절대 오차 계산
        absolute_error = abs(row['remaining_time'] - update_prediction)
        absolute_errors[mixing_ratio].append(absolute_error)
        
        # 정확도 기준: ±5분 이내의 오차
        if absolute_error <= 10:
            correct_predictions[mixing_ratio] += 1
        
        # 데이터 개수 증가
        total_data_count[mixing_ratio] += 1

        # 예측 결과 저장
        predictions.append({
            'temperature': row['temperature'],
            'mixing_ratio': mixing_ratio,
            'refresh_count': row['refresh_count'],
            'elapsed_time': row['elapsed_time'],
            'growth_percentage': row['growth_percentage'],
            'remaining_time': row['remaining_time'],
            f'model_initial[{mixing_ratio}] output': initial_prediction,
            f'model_update[{mixing_ratio}] output': update_prediction
        })

    # DataFrame으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 모델별 정확도 및 MAE 계산
    model_accuracies = {}
    overall_absolute_errors = []
    overall_correct_predictions = 0
    overall_total_data = 0

    print("모델별 정확도 및 MAE:")
    for ratio in absolute_errors.keys():
        mae = np.mean(absolute_errors[ratio])
        accuracy = (correct_predictions[ratio] / total_data_count[ratio]) * 100
        model_accuracies[ratio] = {'MAE': mae, 'Accuracy': accuracy}

        # 출력
        print(f"Mixing Ratio {ratio}:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  정확도 (±10분 이내): {accuracy:.2f}%")
        
        # 전체 데이터 계산
        overall_absolute_errors.extend(absolute_errors[ratio])
        overall_correct_predictions += correct_predictions[ratio]
        overall_total_data += total_data_count[ratio]

    # 전체 정확도 및 MAE 계산
    overall_mae = np.mean(overall_absolute_errors)
    overall_accuracy = (overall_correct_predictions / overall_total_data) * 100

    print("\n전체 정확도 및 MAE:")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.2f}")
    print(f"  전체 정확도 (±10분 이내): {overall_accuracy:.2f}%")

    # CSV 파일로 저장
    predictions_df.to_csv('./res/predictions_output_with_accuracy.csv', index=False)

    return predictions_df, model_accuracies, overall_mae, overall_accuracy

# 실행
predicted_results, model_accuracies, overall_mae, overall_accuracy = predict_remaining_time_with_model_accuracy('./res/filtered_original_data.csv')
