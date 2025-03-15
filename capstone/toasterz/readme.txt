프로젝트 목적:
	발효종의 발효가 완료되는 시간 실시간 예측

데이터 전처리 방식:
	100% 성장 시점을 기준으로 remaining_time을 계산하여 채운다
	우리가 원하는 것은 100%에 도달하는 시점이기 때문에 실험 중 발효종이 100%를 넘어서 성장할 경우 그 이후의 데이터는 삭제한다
	100%에 도달하지 못한 발효종의 경우 remaining_time을 구할 수 없기 때문에 배제한다

데이터 형식:
	temperature,mixing_ratio,refresh_count,elapsed_time,growth_percentage,remaining_time
	24,1:1:1,3,0,0,305
	24,1:1:1,3,3,1,302
	24,1:1:1,3,110,15,195
	...

	temperature = 현재 발효종의 온도
	mixing_ratio = 발효종:밀가루:물의 혼합 비율
	refresh_count = refresh를 한 횟수(발효종의 성장 단계와 유사)
	elapsed_time = 마지막 refresh를 한 시점부터의 경과 시간
	growth_percentage = 마지막 refresh를 한 시점부터의 발효종이 부푼 비율(2배가 된 시점이 100%)
	remaining_time = 발효종이 2배가 될 때까지 남은 시간(지도 학습에 사용될 정답에 해당)

설계 방식:
mixing_ratio는 1:1:1, 1:2:2, 1:5:5, 1:10:10으로 총 4개이며, 각각의 비율마다 다른 모델을 설계한다. 따라서 모델은 총 8개가 설계된다.
1. model_initial
	temperature, refresh_count를 기반으로 remaining_time을 예측한다.
2. model_update
	model_initial의 output과 elapsed_time, growth_percentage를 기반으로 실시간 remaining_time을 예측한다.

즉, model_initial은 발효가 끝날 때까지 걸리는 전체 시간을 계산하고, model_update는 발효 도중 남은 시간을 지속적으로 예측해 사용자에게 보여주는 역할을 하게 된다.
발효종의 성장 속도는 점점 빨라지기 때문에 첨부한 그래프와 같은 형태를 띄게 된다.

현재 문제점:
	model_update의 예측 값의 범위가 [0, model_initial의 output]을 지키지 못하고 있음 (0% 성장 시 남은 시간이 model_initial의 output과 일치해야 하고, 100% 성장 시 남은 시간이 0이어야 하지만 그렇지 못함)

*현재 학습에 사용한 데이터는 ideal_synthetic_fermentation_data_v4.csv로 프로그램으로 만들어낸 가짜 데이터이다.