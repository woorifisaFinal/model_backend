import pandas as pd
import numpy as np
from stage1 import getPrediction



def runAlpha():
    
    df_3_months = getPrediction()

    # 포트폴리오 생성을 위한 자산 리스트
    assets = df_3_months.columns

    # 포트폴리오 생성 횟수
    num_portfolios = 10000

    port_ratios = []  # 포트폴리오 비중 리스트
    port_returns = np.array([])  # 연간 수익률 배열
    port_risks = np.array([])  # 연간 리스크(변동성) 배열

    lookback_period_days = 90  # 3개월치 데이터를 사용하고 싶을 때 설정 (1일당 1개 데이터를 가정)
    
    for i in range(num_portfolios):
        # 무작위로 포트폴리오 비중 생성
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)  # 총합을 1로 만들어줌

        portfolio_returns = np.sum(weights * df_3_months, axis=1)

        # 포트폴리오의 3개월간 누적 수익률 계산
        cumulative_returns = np.prod(1 + portfolio_returns) - 1

        # 포트폴리오 리스크 계산 (간단하게 표준편차를 사용)
        cov_matrix = df_3_months.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        port_ratios.append(weights)
        port_returns = np.append(port_returns, cumulative_returns)
        port_risks = np.append(port_risks, portfolio_volatility)


    # 결과 데이터프레임 생성
    portfolio_data = {'Returns': port_returns, 'Risk': port_risks}
    for i, asset in enumerate(assets):
        portfolio_data[asset] = [weights[i] for weights in port_ratios]

    portfolio_df = pd.DataFrame(portfolio_data)


    # 최대 샤프 지수를 갖는 포트폴리오 선택
    max_sharpe_portfolio = portfolio_df.iloc[portfolio_df['Returns'].idxmax()]
    max_sharpe_portfolio['type'] = '공격형'

    # 최소 리스크를 갖는 포트폴리오 선택
    min_risk_portfolio = portfolio_df.iloc[portfolio_df['Risk'].idxmin()]
    min_risk_portfolio['type'] = '안정형'

    return max_sharpe_portfolio[2:], min_risk_portfolio[2:]