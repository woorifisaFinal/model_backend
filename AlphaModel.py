import pandas as pd
import numpy as np
from stage1 import getPrediction
import datetime


def alpha_model():
    def runAlpha():
        
        df_1_months = getPrediction()
        # 한달만

        # 포트폴리오 생성을 위한 자산 리스트
        assets = df_1_months.columns

        # 포트폴리오 생성 횟수
        num_portfolios = 10000

        port_ratios = []  # 포트폴리오 비중 리스트
        port_returns = np.array([])  # 연간 수익률 배열
        port_risks = np.array([])  # 연간 리스크(변동성) 배열


        
        for i in range(num_portfolios):
            # 무작위로 포트폴리오 비중 생성
            weights = np.random.random(len(assets))
            weights /= np.sum(weights)  # 총합을 1로 만들어줌

            portfolio_returns = np.sum(weights * df_1_months, axis=1)

            # 포트폴리오의 1개월간 누적 수익률 계산
            monthly_returns = portfolio_returns + 1
            cumulative_returns_1_month = np.prod(monthly_returns) - 1

            # 포트폴리오 리스크 계산 (간단하게 표준편차를 사용)
            cov_matrix = df_1_months.cov()
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            port_ratios.append(weights)
            port_returns = np.append(port_returns, cumulative_returns_1_month)
            port_risks = np.append(port_risks, portfolio_volatility)


        # 결과 데이터프레임 생성
        portfolio_data = {'Returns': port_returns, 'Risk': port_risks}
        for i, asset in enumerate(assets):
            portfolio_data[asset] = [weights[i] for weights in port_ratios]

        portfolio_df = pd.DataFrame(portfolio_data)


        # 최대 샤프 지수를 갖는 포트폴리오 선택
        max_sharpe_portfolio = portfolio_df.iloc[portfolio_df['Returns'].idxmax()]
        return max_sharpe_portfolio


    def wish_date_weight(): # date는 '$$$$-$$-$$'형식으로 받아온다
        result = runAlpha()
        stable_asset = ['kor3y','kor10y','us3y','us10y','gold']
        risky_asset = ['us', 'uk' ,'jp',	'euro',	 'ind',	'tw',	'br','kor']

        stable_weight = pd.DataFrame(columns=['kor3y','kor10y','us3y','us10y','gold'])
        stable_sum = 0
        risky_weight = pd.DataFrame(columns=['us', 'uk' ,'jp',	'euro',	 'ind',	'tw',	'br','kor'])
        risky_sum = 0

        row = result.copy()
        stable_sum = row[stable_asset].sum()
        risky_sum =  row[risky_asset].sum()

        final = row.copy() #result 원본값을 그대로 보존하기 위해서
        if stable_sum<0.4:

            risky_weight= row.copy() # 모델의 가중치가 곧 공격형 자산비중
            # print( risky_weight - result)
            final[stable_asset] = row[stable_asset]*(0.4/stable_sum)

            final[risky_asset] = row[risky_asset]*(0.6/risky_sum)
            
            stable_weight = final # 수정한 비중으로 안전형 자산

        elif stable_sum>=0.4: 

            stable_weight = row.copy() # 모델의 가중치가 곧 안전형 자산 비중

            final[stable_asset] = row[stable_asset]*(0.4/stable_sum)
            
            final[risky_asset] = row[risky_asset]*(0.6/risky_sum)
            
            risky_weight = final # 수정한 비중으로 공격형 자산
        return stable_weight, risky_weight

    stable_weight , risky_weight = wish_date_weight()
    # stable_weight = stable_weight.astype('float')
    # stable_weight['type'] = "A/안정형"
    # stable_weight['type'] = stable_weight['type'].astype(str)
    
  
    # risky_weight = risky_weight.astype('float')
    # risky_weight['type'] = "A/공격형"
    # risky_weight['type'] = risky_weight['type'].astype(str)
    
    return stable_weight[2:], risky_weight[2:]

