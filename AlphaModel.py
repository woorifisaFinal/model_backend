import pandas as pd
import numpy as np



def runAlpha():
    data = pd.read_csv("psuedo_2022_prediction.csv")

    col_list = ['us', 'uk', 'jp', 'euro', 'kor', 'ind', 'tw', 'br', 'kor3y', 'kor10y', 'us3y', 'us10y', 'gold'] 

    rename_dict = {
        'kospi':"kor",
        'nasdaq':"us",
        'euro_stoxx':"euro",
        'ftse':"uk",
        'nikkei':"jp",
        'korea_bond_03':"kor3y",
        'korea_bond_10':"kor10y",
        'america_bond_03':"us3y",
        'america_bond_10':"us10y",
        'gold':"gold",
        'brazil':"br",
        'taiwan':"tw",
        'india':"ind"
        }

    data.set_index("date", inplace=True)
    data.rename(columns=rename_dict, inplace=True)
    data = data[col_list]


    # 기간 설정
    lookback_period_months = 3  # 3개월치 데이터를 사용하고 싶을 때 설정

    # 기간 계산
    end_date = pd.Timestamp(2022,12,31)
    start_date = end_date - pd.DateOffset(months=lookback_period_months)

    # 데이터 추출
    df_3_months = data.loc[(data.index > start_date.strftime('%Y-%m-%d')) & (data.index <= end_date.strftime('%Y-%m-%d'))]


    # 포트폴리오 생성을 위한 자산 리스트
    assets = data.columns

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