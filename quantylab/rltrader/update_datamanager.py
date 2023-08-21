import os
import pandas as pd
import numpy as np
import threading
import os
import pandas as pd
import numpy as np
import ta
import scipy
import pymysql
import talib
import time
from sklearn.preprocessing import RobustScaler
from joblib import dump, load
from pandas.tseries.offsets import *
from datetime import datetime
from quantylab.rltrader import settings
from quantylab.rltrader.redis_auto import redis_self
from sklearn.preprocessing import RobustScaler
from joblib import dump, load
from tqdm import tqdm
import schedule

    
COLUMNS_CHART_DATA = ['date_time', 'open', 'high', 'low', 'close', 'volume',
'Accumulated sales volume', 'Accumulated number of contract']


COLUMNS_TRAINING_DATA_V4 = ['MFI',
       'ADI', 'OBV', 'CMF', 'FI', 'EOM, EMV', 'VPT', 'NVI', 'VMAP', 'ATR',
       'BHB', 'BLB', 'KCH', 'KCL', 'KCM', 'DCH', 'DCL', 'DCM', 'UI', 'WMA',
       'MACD', 'ADX', 'MINUS_VI', 'PLUS_VI', 'TRIX', 'MI', 'CCI', 'DPO', 'KST',
       'Ichimoku', 'Parabolic SAR', 'STC', 'RSI', 'SRSI', 'TSI', 'UO', 'SR',
       'WR', 'AO', 'KAMA', 'ROC', 'PPO', 'PVO', 'close yesterday',
       'ratio high close', 'ratio low close', 'ratio open close', 'mdd', 'MA5',
       'SMA5 VOLUME', 'ratio close MA5', 'MA10', 'SMA10 VOLUME',
       'ratio close MA10', 'MA20', 'SMA20 VOLUME', 'ratio close MA20', 'MA60',
       'SMA60 VOLUME', 'ratio close MA60', 'MA120', 'SMA120 VOLUME',
       'ratio close MA120', 'EMA5', 'EMA5 VOLUME', 'EMA12', 'EMA12 VOLUME',
       'EMA20', 'EMA20 VOLUME', 'EMA30', 'EMA30 VOLUME', 'EMA90',
       'EMA90 VOLUME', 'change', 'peak_diffratio', 'interpeak_mdd', 'news',
       'Exchange Rate', 'kospi close', 'kospi close/MA5', 'kospi close/MA10',
       'kospi close/MA20', 'kospi close/MA60', 'kospi close/MA120',
       'kospi close/EMA5', 'kospi close/EMA12', 'kospi close/EMA20',
       'kospi close/EMA30', 'kospi close/EMA90']

def clear_lists():
    tick_list.clear()
    kospi_list.clear()
    news_list.clear()
    stock = pd.DataFrame()
    
    
# Redis로부터 데이터를 구독하고 리스트에 숫자를 추가하는 함수
def receive_tick(result_list, stop_event):
    while not stop_event.is_set():
        data = redis_self.sub_msg('1')  # Redis에서 'modeltick' 키로부터 데이터를 수신합니다. 
        result_list.append(data)  # 수신한 데이터를 'tick_list'에 추가합니다.

def receive_kospi(result_list, stop_event):
    while not stop_event.is_set():
        data = redis_self.sub_msg('kospii')  # Redis에서 'tickmodel' 키로부터 데이터를 수신합니다.
        result_list.append(data)  # 수신한 데이터를 'kospi_list'에 추가합니다.
        print('+___________________+')

def receive_news(result_list, stop_event):
    while not stop_event.is_set():
        data = redis_self.sub_msg('3')  # Redis에서 'modelnews' 키로부터 데이터를 수신합니다.
        result_list.append(data)  # 수신한 데이터를 'news_list'에 추가합니다.
        
stop_event = threading.Event()

def load_data_v3_v4(code, ver):
    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3
    elif ver == 'v4':
        columns = COLUMNS_TRAINING_DATA_V4
        
    # Redis로부터 수신한 데이터를 저장할 빈 리스트들을 생성합니다.
    tick_list = []
    kospi_list = []
    news_list = []

    # Redis로부터 데이터를 받아올 별도의 스레드들을 생성합니다.
    thread_tick = threading.Thread(target=receive_tick, args=(tick_list, stop_event))
    thread_kospi = threading.Thread(target=receive_kospi, args=(kospi_list, stop_event))
    thread_news = threading.Thread(target=receive_news, args=(news_list, stop_event))

    # 스레드들을 시작합니다.
    thread_tick.start()
    thread_kospi.start()
    thread_news.start()

    while True:
        current_time = datetime.now().time()
        if 9 <= current_time.hour < 15 or (current_time.hour == 15 and current_time.minute <= 20):
            print('틱 데이터 {}개'.format(tick_list))       
        else:
            break

    stop_event.set()
    thread_tick.join()  # Wait for the thread_tick to finish
    thread_kospi.join() # Wait for the thread_kospi to finish
    thread_news.join()

    chart_data = pd.DataFrame()
    
    training_data = pd.DataFrame()
    
    stock = pd.DataFrame(tick_list, columns=['date_time', 'open', 'high', 'low', 'close', 'volume', 'Accumulated sales volume', 'Accumulated number of contract'])

    if kospi_list:
        kospi = pd.DataFrame(kospi_list, columns=['kospi_date_time', 'kospi close','Exchange Rate'])
        stock = pd.concat([stock, kospi], axis=1)

    if news_list:
        news = pd.DataFrame(news_list, columns=['news_date_time', 'news'])
        stock = pd.concat([stock, news], axis=1)


    # nan 값을 이전 데이터로 채웁니다.
    stock.fillna(method='ffill', inplace=True)
    stock.drop(['kospi_date_time','news_date_time'], axis=1, inplace=True)

    stock['Exchange Rate'] = stock['Exchange Rate'].str.replace(',','',regex=False)
    stock['kospi close'] = stock['kospi close'].str.replace(',','',regex=False)

    stock = stock.astype({'open':'float', 'high':'float', 'low':'float', 'close':'float', 'volume':'float', 'Accumulated sales volume':'float', 'Accumulated number of contract':'float', 'Exchange Rate':'float','kospi close':'float'})

    H, L, C, V = stock['high'], stock['low'], stock['close'], stock['volume']

    stock['MFI'] = ta.volume.money_flow_index(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock['ADI'] = ta.volume.acc_dist_index(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock['OBV'] = ta.volume.on_balance_volume(close=C, volume=V, fillna=True)
    stock['CMF'] = ta.volume.chaikin_money_flow(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock['FI'] = ta.volume.force_index(close=C, volume=V, fillna=True)
    stock['EOM, EMV'] = ta.volume.ease_of_movement(
        high=H, low=L, volume=V, fillna=True)

    stock['VPT'] = ta.volume.volume_price_trend(close=C, volume=V, fillna=True)
    stock['NVI'] = ta.volume.negative_volume_index(close=C, volume=V, fillna=True)
    stock['VMAP'] = ta.volume.volume_weighted_average_price(
        high=H, low=L, close=C, volume=V, fillna=True)

    # Volatility
    stock['ATR'] = ta.volatility.average_true_range(
        high=H, low=L, close=C, fillna=True)
    stock['BHB'] = ta.volatility.bollinger_hband(close=C, fillna=True)
    stock['BLB'] = ta.volatility.bollinger_lband(close=C, fillna=True)
    stock['KCH'] = ta.volatility.keltner_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock['KCL'] = ta.volatility.keltner_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock['KCM'] = ta.volatility.keltner_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock['DCH'] = ta.volatility.donchian_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock['DCL'] = ta.volatility.donchian_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock['DCM'] = ta.volatility.donchian_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock['UI'] = ta.volatility.ulcer_index(close=C, fillna=True)
    # Trend
    stock['WMA'] = ta.trend.wma_indicator(close=C, fillna=True)
    stock['MACD'] = ta.trend.macd(close=C, fillna=True)
    stock['ADX'] = ta.trend.adx(high=H, low=L, close=C, fillna=True)
    stock['MINUS_VI'] = ta.trend.vortex_indicator_neg(
        high=H, low=L, close=C, fillna=True)
    stock['PLUS_VI'] = ta.trend.vortex_indicator_pos(
        high=H, low=L, close=C, fillna=True)
    stock['TRIX'] = ta.trend.trix(close=C, fillna=True)
    stock['MI'] = ta.trend.mass_index(high=H, low=L, fillna=True)
    stock['CCI'] = ta.trend.cci(high=H, low=L, close=C, fillna=True)
    stock['DPO'] = ta.trend.dpo(close=C, fillna=True)
    stock['KST'] = ta.trend.kst(close=C, fillna=True)
    stock['Ichimoku'] = ta.trend.ichimoku_a(high=H, low=L, fillna=True)
    stock['Parabolic SAR'] = ta.trend.psar_down(
        high=H, low=L, close=C, fillna=True)
    stock['STC'] = ta.trend.stc(close=C, fillna=True)
    # Momentum
    stock['RSI'] = ta.momentum.rsi(close=C, fillna=True)
    stock['SRSI'] = ta.momentum.stochrsi(close=C, fillna=True)
    stock['TSI'] = ta.momentum.tsi(close=C, fillna=True)
    stock['UO'] = ta.momentum.ultimate_oscillator(
        high=H, low=L, close=C, fillna=True)
    stock['SR'] = ta.momentum.stoch(close=C, high=H, low=L, fillna=True)
    stock['WR'] = ta.momentum.williams_r(high=H, low=L, close=C, fillna=True)
    stock['AO'] = ta.momentum.awesome_oscillator(high=H, low=L, fillna=True)
    stock['KAMA'] = ta.momentum.kama(close=C, fillna=True)
    stock['ROC'] = ta.momentum.roc(close=C, fillna=True)
    stock['PPO'] = ta.momentum.ppo(close=C, fillna=True)
    stock['PVO'] = ta.momentum.pvo(volume=V, fillna=True)

    #등락률 계산
    stock["close yesterday"] = stock["close"].shift(1)
    stock["change"] = ((stock["close"] - stock["close yesterday"]) / stock["close yesterday"])*100
    stock["ratio high close"] = ((stock["high"] - stock["close yesterday"]) / stock["close yesterday"])*100
    stock["ratio low close"] = ((stock["low"] - stock["close yesterday"]) / stock["close yesterday"])*100
    stock["ratio open close"] = (stock["open"] - stock["close yesterday"]) / stock["close yesterday"]*100
    stock.drop("close yesterday", axis=1)

    #mdd 구하는 것
    window = 380 #하루 분봉데이터 개수
    #종가에서 하루 단위 최고치 peak를 구함
    peak = stock['close'].rolling(window, min_periods=1).max()
    # 3.최고치 대비 현재 종가가 얼마나 하락했는지 구함
    drawdown = stock['close']/peak - 1.0
    # 4. drawdown에서 1년기간 단위로 최저치 max_dd를 구한다. 
    stock['mdd'] = drawdown.rolling(window, min_periods=1).min()
    #mdd가 낮을수록 위험성이 적은 것이다

    periods = [5, 10, 20, 60, 120]

    for period in periods:
        stock[f'MA{period}'] = talib.SMA(stock['close'], period)
        stock[f'SMA{period} VOLUME'] = talib.SMA(stock['volume'], period)
        stock[f'ratio close MA{period}'] = stock["close"] / stock[f"MA{period}"]
        stock[f'kospi close/MA{period}'] = talib.SMA(stock['kospi close'], period)

    periods2 = [5, 12, 20, 30, 90]
    for period2 in periods2:
        stock[f'EMA{period2}'] = talib.EMA(stock['close'], period2)
        stock[f'EMA{period2} VOLUME'] = talib.EMA(stock['volume'], period2)
        stock[f'kospi close/EMA{period2}'] = talib.EMA(stock['kospi close'], period2)


    # peaks 리스트 초기화
    peaks = []
    # 양수 peak 찾기
    peaks.extend(scipy.signal.find_peaks(stock['close'], distance=5, width=10)[0])

    # 음수 peak 찾기
    peaks.extend(scipy.signal.find_peaks(-stock['close'], distance=5, width=10)[0])

    # 마지막 날짜가 peak인 경우 peaks에 추가
    if len(stock)-1 not in peaks:
        peaks.append(len(stock)-1)

    # 수익성, 안전성, 유동성 지표 추가
    stock.loc[:, 'peak_date'] = ''
    stock.loc[:, 'peak_close'] = np.nan
    stock.loc[:, 'peak_diffratio'] = np.nan
    stock.loc[:, 'interpeak_mdd'] = np.nan
#           stock.loc[:, 'interpeak_trans_price_exp'] = np.nan
#           stock.loc[:, 'trans_price_exp']=np.nan

    _last_date = stock['date_time'][0] # 이전 날짜를 저장하는 변수 초기화
    for peak in peaks:  # peaks 리스트의 각 peak에 대해 반복
        _date = stock.iloc[peak]['date_time']  # 현재 peak의 날짜 저장
        _close = stock.iloc[peak]['close']  # 현재 peak의 종가 저장
        mask = (stock['date_time'] >= _last_date) & (stock['date_time'] <= _date)  # 이전 날짜부터 현재 peak 날짜까지의 mask 생성
        _last_date = _date  # 이전 날짜를 현재 peak의 날짜로 업데이트
        if len(stock[mask]) > 0:  # mask에 해당하는 데이터가 있는 경우
            stock.loc[mask, 'peak_date'] = _date  # mask에 해당하는 행의 'peak_date' 열에 현재 peak의 날짜 저장
            stock.loc[mask, 'peak_close'] = _close  # mask에 해당하는 행의 'peak_close' 열에 현재 peak의 종가 저장
            _x = np.array(stock.loc[mask, 'close'])  # mask에 해당하는 행의 'close' 열을 numpy 배열로 변환하여 _x에 저장
            lower = np.argmax(np.maximum.accumulate(_x) - _x)  # 최대값을 누적한 후 현재 값과의 차이 중 최대값의 인덱스를 lower에 저장
            upper = np.argmax(_x[:lower+1])  # lower 이전까지의 값 중 최대값의 인덱스를 upper에 저장
            stock.loc[mask, 'interpeak_mdd'] = (_x[lower] - _x[upper]) / _x[upper]  # mask에 해당하는 행의 'interpeak_mdd' 열에 MDD 계산 결과 저장
            # stock.loc[mask, 'interpeak_trans_price_exp'] = stock.loc[mask, 'trans_price_exp'].mean()  # mask에 해당하는 행의 'interpeak_trans_price_exp' 열에 'trans_price_exp' 열의 평균값 저장 = nan값만 나온다

    stock.loc[:, 'peak_diffratio'] = (stock.loc[:, 'peak_close'] - stock.loc[:, 'close']) / stock.loc[:, 'close']  # 'peak_diffratio' 열에 종가와 peak 종가의 차이 비율 계산하여 저장

    stock['date_time'] = stock['date_time'].str.replace(',','',regex=False)
    stock['date_time'] = stock['date_time'].str.replace(':','',regex=False)
    stock['date_time'] = stock['date_time'].str.replace(' ','',regex=False)
    
    
    # stock.drop(range(1, 130), inplace=True) # ~129
    #stock.fillna(method='bfill', inplace=True)
    stock = stock.dropna()
    #tock.drop(['interpeak_trans_price_exp', 'trans_price_exp'], inplace=True, axis=1)

    #차트 보조지표 뉴스 환율 코스피 : 총 97개 지표
    stock.reset_index(inplace=True)
    stock = stock[['date_time', 'open', 'high', 'low', 'close', 'volume',
                    'Accumulated sales volume', 'Accumulated number of contract', 'MFI',
                    'ADI', 'OBV', 'CMF', 'FI', 'EOM, EMV', 'VPT', 'NVI', 'VMAP', 'ATR',
                    'BHB', 'BLB', 'KCH', 'KCL', 'KCM', 'DCH', 'DCL', 'DCM', 'UI', 'WMA',
                    'MACD', 'ADX', 'MINUS_VI', 'PLUS_VI', 'TRIX', 'MI', 'CCI', 'DPO', 'KST',
                    'Ichimoku', 'Parabolic SAR', 'STC', 'RSI', 'SRSI', 'TSI', 'UO', 'SR',
                    'WR', 'AO', 'KAMA', 'ROC', 'PPO', 'PVO', 'close yesterday',
                    'ratio high close', 'ratio low close', 'ratio open close', 'mdd', 'MA5',
                    'SMA5 VOLUME', 'ratio close MA5', 'MA10', 'SMA10 VOLUME',
                    'ratio close MA10', 'MA20', 'SMA20 VOLUME', 'ratio close MA20', 'MA60',
                    'SMA60 VOLUME', 'ratio close MA60', 'MA120', 'SMA120 VOLUME',
                    'ratio close MA120', 'EMA5', 'EMA5 VOLUME', 'EMA12', 'EMA12 VOLUME',
                    'EMA20', 'EMA20 VOLUME', 'EMA30', 'EMA30 VOLUME', 'EMA90',
                    'EMA90 VOLUME', 'change', 'peak_diffratio', 'interpeak_mdd', 'news',
                    'Exchange Rate', 'kospi close', 'kospi close/MA5', 'kospi close/MA10',
                    'kospi close/MA20', 'kospi close/MA60', 'kospi close/MA120',
                    'kospi close/EMA5', 'kospi close/EMA12', 'kospi close/EMA20',
                    'kospi close/EMA30', 'kospi close/EMA90']]


    
    # 처리한 데이터는 리스트에서 삭제합니다.
    schedule.every().day.at("15:25").do(clear_lists)
    print('삭제 : ',stock)

    chart_data = stock[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = stock[columns].values
    
    # 스탑이벤트 재조정으로 다음 함수 호출때 쓰레드 재가동
    stop_event.clear()
    
    
    # 스케일링
    if ver == 'v4':
        scaler_path = os.path.join(settings.BASE_DIR, 'scalers', f'scaler_{ver}.joblib')
        scaler = None
        if not os.path.exists(scaler_path):
            scaler = RobustScaler()
            scaler.fit(training_data)
            dump(scaler, scaler_path)
        else:
            scaler = load(scaler_path)
        training_data = scaler.transform(training_data)
    
    return chart_data, training_data