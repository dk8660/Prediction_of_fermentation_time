import pandas as pd
import joblib

# 하나의 입력에 대해 단순하게 예측 결과만을 보여주는 코드

def predict_single_entry(temperature, mixing_ratio, refresh_count, elapsed_time, growth_percentage):
    # model_initial 예측
    model_initial_filename = f'./model/model_initial_{mixing_ratio.replace(":", "")}.joblib'
    model_initial = joblib.load(model_initial_filename)
    
    initial_features = pd.DataFrame([[temperature, refresh_count]], columns=['temperature', 'refresh_count'])
    initial_prediction = model_initial.predict(initial_features)[0]

    # model_update 예측
    model_update_filename = f'./model/model_update_{mixing_ratio.replace(":", "")}.joblib'
    model_update = joblib.load(model_update_filename)
    
    update_features = pd.DataFrame(
        [[initial_prediction, elapsed_time, growth_percentage]], 
        columns=['initial_prediction', 'elapsed_time', 'growth_percentage']
    )
    update_prediction = int(round(model_update.predict(update_features)[0]))

    # 강제 조정 적용
    if growth_percentage == 100:
        update_prediction = 0  # 성장 비율이 100이면 무조건 0
    elif growth_percentage < 100 and update_prediction == 0:
        update_prediction = 1  # 성장 비율이 100이 아닌데 0이면 1로 조정

    # 예측 결과 반환
    result = update_prediction
    
    return result

# 예제 실행
if __name__ == "__main__":
    # 테스트 입력 예시
    result = predict_single_entry(
        temperature=24,
        mixing_ratio='1:1:1',
        refresh_count=3,
        elapsed_time=110,
        growth_percentage=15
    )
    print(result)
