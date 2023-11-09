# AI 주식 트레이딩 프로그램
---
- 주식 데이터를 모델에 학습시켜 실시간으로 거래하고, 결과에 따라 자동으로 모델 성능을 개선하는 프로그램
- 구글 클라우드를 이용해 계속해서 프로그램 실행
- 주가 데이터 + 시장 데이터(코스피, 환율) + 여론 데이터(종목 관련 뉴스) 고려하여 주가 예측후 실시간 매매


# 프로젝트 아키텍쳐
![image](https://github.com/jyy4014-git/ml-dev/assets/134044918/22ad81a0-eafd-4584-9480-0c31146e7413)


# 강화학습 설계도
![image](https://github.com/jyy4014-git/ml-dev/assets/134044918/378628e4-0c91-498f-9c16-8c86e6796ea4)

main 실행 -> data_manager.py 로 데이터 수집 및 처리 -> 환경모듈에 종가 반환 -> 학습기 모듈에서 A3C 선택 -> networks.py의 pytorch 신경망(CNN) 선택 -> agent에서 매매를 실현하며 reward로 모델 학습 -> visualizer.py 로 모델링 성능 체크

# Learners 과정
![image](https://github.com/jyy4014-git/ml-dev/assets/134044918/9635157a-ab26-4cc1-a10d-5867ef281b0e)
1. 모델링 초기에 Epsilon 1을 주어 탐험 100% 이며 회차가 거듭될수록 탐험 비율이 점점 줄어든다
2. 탐험은 무작위로 매수/매도를 수행하며 결과에 지연보상(수익률이 양수)이 발생한 경우 배치 학습데이터를 생성해 신경망을 업데이트 한다
3. 탐험해서 지연보상이 발생하지 않는 경우 첫 단계로 돌아간다. 에포크가 종료되면 환경을 초기화하여 Epsilon 비율에 따라탐험/신경망 행동 중 결정한다
4. 신경망 학습으로 매수/매도를 결정하고 지연보상이 발생하면 신경망 업데이트한다. 그 외엔 에포크 진행 여부에 따라 탐험/신경망 행동 중 결정한다
5. 강화학습 종료 후 나오는 그래프로 모델의 성능을 확인하며 파라미터 조절한다

# 프로그램 소스코드
- main.py로 프로그램을 실행
- data_manager.py 에서 Redis로 실시간 주식 데이터 연동 및 전처리까지 완료
- ml-dev/quantylab/rltrader/networks/networks_pytorch.py 에서 cnn 신경망 구성
- quantylab/rltrader/agent.py 에서 학습된 모델이 매수/매도/홀딩 한다
- 하루의 거래 결과가 좋지 않은 경우 quantylab/rltrader/learners.py 에서 가치신경망, 정책신경망을 a3c로 학습
