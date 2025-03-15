import pandas as pd
import joblib
import numpy as np

# csv 파일을 이용하여 많은 데이터를 한 번에 예측시켜 결과를 얻을 수 있게한 코드

def predict_remaining_time(input_csv):
    data = pd.read_csv(input_csv)
    predictions = []
    absolute_errors = []
    correct_predictions = 0

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

        # 절대 오차 계산 및 정확도 체크
        absolute_error = abs(row['remaining_time'] - update_prediction)
        absolute_errors.append(absolute_error)
        
        # 정확도 기준: ±5분 이내의 오차
        if absolute_error <= 5:
            correct_predictions += 1

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

    # 정확도 측정
    mae = np.mean(absolute_errors)
    accuracy = (correct_predictions / len(data)) * 100

    # 결과 출력
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"차이가 5분 내인 경우: {accuracy:.2f}%")

    # CSV 파일로 저장
    predictions_df.to_csv('./res/predictions_output.csv', index=False)
    return predictions_df, mae, accuracy

# 예측 및 정확도 측정 실행
predicted_results, mae, accuracy = predict_remaining_time('./res/structured_data.csv') # CSV 경로
print(predicted_results)
