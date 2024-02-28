# AI 주식 HELPER
<br><br>
![ezgif com-optimize](https://github.com/jyy4014-git/ml-dev/assets/134044918/8fdeb8b6-5907-4891-9c96-89078a2c744c)
![강화학습 웹](https://github.com/jyy4014-git/ml-dev/assets/134044918/3cde271a-73b2-44de-9c22-097e5b96fb51)



## 서비스
- AI 투자 : AI 실시간 매수 매도 서비스
- 종목정보 : 실시간 주가, 주가 그래프, 뉴스, 재무지표 등 투자 기초 자료 제공
<br><br>

# 백엔드 기술 스택
- Language: java,python
- Framework: Spring, SpringBoot, Pandas, Pytorch, Numpy
- Build Tool: Gradle
- DB: MySQL, Redis
- Server: GCP
- Other Tools : Git, Github, notion, Google Cloud, Anaconda, Jupyter, VScode, DaeshinAPI, 
<br><br>

# 프로젝트 아키텍쳐
![image](https://github.com/jyy4014-git/ml-dev/assets/134044918/f74bd81a-c452-4c3d-b5c5-f78062e895c7)

<br><br>

# 강화학습 설계도
![image](https://github.com/jyy4014-git/ml-dev/assets/134044918/378628e4-0c91-498f-9c16-8c86e6796ea4)


<br><br>

# Trouble Shooting
### 주식 데이터 저장 및 조회
- 기존 분봉 데이터 2년치 약 18만5천개를 받아올 때 API 호출제한으로 1시간 넘게 걸리던 것을 호출 지연 및 자동화 설계로 25초만에 줄임
- 주식 시장에서는 거래가 없는 시간대에 대한 데이터 불일치로 인해 NaN 값이 발생하여 신경망 모델 학습시키는데 문제를 일으키는 현상 발생. 전처리 후 유DB저장하여 데이터 일관성과 정확성을 높이고 신경망 모델의 학습 효율성 향상

### 데이터 전처리
- KOSPI, 환율, 뉴스 감성 분석 데이터는 24시간 수집되어 주식 데이터와 시간대가 안맞는 문제 발생. 데이터 수집은 24시간으로 하되 전처리 후 시간대는 주식데이터에 맞추어 신경망에 같이 학습할 수 있게 만듬
- 경제 용어 감성분석은 일반 자연어 처리와 의미가 다르기 때문에 정확도가 매우 낮게 나옴. 뉴스데이터 감성 데이터와 KOBERT 모델을 이용해 정확도 85% 로 향상
- 주식 데이터에 있는 이상치들 때문에 정확도가 낮게 나옴. ROBUST 스케일링으로 정확도 향상

### 실시간 데이터 파이프라인 설계 
- 시간대가 다른 데이터 소스(주식, 뉴스, KOSPI)에서 수집된 데이터를 비동기 방식으로 수집 및 전처리함으로써 데이터 처리 시간을 최대 30%까지 단축
- DB에서 데이터 조회후 예측하고 실시간 매매하던 것을 REDIS를 이용하여 매매 시간 단축으로 거래속도에 우위 선점
- 실시간 주가 내용을 웹에 표시할때 새로고침 없이 데이터 갱신하기 위해 5초마다 DB의 데이터 조회하여 웹에 표시하게 함

