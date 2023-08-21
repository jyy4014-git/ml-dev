import os
import sys
import logging
import argparse
import json
import time
import datetime
import threading
import pandas as pd
os.environ['RLTRADER_BASE'] = os.path.dirname(os.path.abspath(__file__))
from quantylab.rltrader.redis_auto import redis_self
from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager


    

def clear_lists():
    daily_chart_data.clear()
    daily_training_data.clear()
               
def real_clear():
    list_stock_code.clear()
    list_chart_data.clear()
    list_training_data.clear()
    list_min_trading_price.clear()
    list_max_trading_price.clear()
        
        
if __name__ == '__main__':
    
    print('처음시작')
    stock = redis_self.sub_msg('channel1')
    initial_balance = stock['주문 가능 금액']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='predict')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v4')
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--stock_code', default='012450')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default='a3c')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='cnn')
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--balance', type=int, default=initial_balance)
    args = parser.parse_args()


    


    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 100 if args.mode in ['train', 'update'] else 1
    num_steps = 30 if args.net in ['lstm', 'cnn'] else 1
    

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
    


    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)
    


    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    

    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []
    
    daily_chart_data = []
    daily_training_data = []


    

    stock_code = args.stock_code
    #         for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
    data = redis_self.sub_msg('4')
    real_profitloss = data['수익률']
    initial_profit = real_profitloss
    
    while True:
        stock = redis_self.sub_msg('channel1')
        initial_balance = stock['주문 가능 금액']
        
        
        chart_data = pd.DataFrame()

        training_data = pd.DataFrame()

        
        chart_data, training_data = data_manager.load_data_v3_v4(
            stock_code, ver=args.ver)

        assert len(chart_data) >= num_steps
        print('데이터수집완료')

        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        
        
        data = redis_self.sub_msg('4')
        real_profitloss = data['수익률']
        real_balance = data['D+2 예상예수금']

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': initial_balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        print('강화학습 시작')
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)
            
            
            daily_chart_data.append(chart_data)
            daily_training_data.append(training_data)
            

        if args.rl_method == 'a3c':
            learner = A3CLearner(**{
                **common_params, 
                'list_stock_code': list_stock_code,
                'list_chart_data': list_chart_data, 
                'list_training_data': list_training_data,
                'list_min_trading_price': list_min_trading_price, 
                'list_max_trading_price': list_max_trading_price,
                'value_network_path': value_network_path, 
                'policy_network_path': policy_network_path})

        assert learner is not None
        
        current_time = datetime.datetime.now().time()
        

        if datetime.time(9, 0) <= current_time <= datetime.time(15, 20):
            print('예측 시작')
            learner.predict()
            print('예측 종료')
            
            real_clear()

            
        else:
            print('예측 시간이 아닙니다')
            real_clear()
            
        
        if not (datetime.time(9, 0) <= current_time <= datetime.time(15, 20)):
            
            if initial_profit > real_profitloss:
                print('추가학습시작')
                learner = A3CLearner(**{
                    **common_params, 
                    'list_stock_code': list_stock_code,
                    'list_chart_data': daily_chart_data, 
                    'list_training_data': daily_training_data,
                    'list_min_trading_price': min_trading_price, 
                    'list_max_trading_price': max_trading_price,
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})


                print('학습시작')
                learner.run()
                learner.save_models()
            clear_lists()
        



