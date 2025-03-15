import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정 (GUI 불필요)
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def predict_and_plot(input_csv):
    data = pd.read_csv(input_csv)
    true_values = []
    predicted_values = []

    for index, row in data.iterrows():
        mixing_ratio = row['mixing_ratio']

        # model_initial 예측
        model_initial_filename = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
        model_initial = joblib.load(model_initial_filename)

        initial_features = pd.DataFrame([[row['temperature'], row['refresh_count']]],
                                        columns=['temperature', 'refresh_count'])
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
            update_prediction = 0
        elif row['growth_percentage'] < 100 and update_prediction == 0:
            update_prediction = 1

        # 저장
        true_values.append(row['remaining_time'])
        predicted_values.append(update_prediction)

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, color='blue', label='Predicted vs True')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', label='Perfect Prediction')
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values (remaining_time)')
    plt.ylabel('Predicted Values (model_update output)')
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    plt.savefig('./comparison_graph.png')
    print("그래프가 './comparison_graph.png'에 저장되었습니다.")

# 실행
if __name__ == "__main__":
    predict_and_plot('./res/filtered_original_data.csv')  # CSV 경로
