import pandas as pd
import yfinance as yf
import sys
import pymysql
from sqlalchemy import create_engine
import datetime

import logging


host = 'team5-db.cvqn3ewwzknb.ap-northeast-2.rds.amazonaws.com'
username = 'admin'
password = 'woorifisa'
## test 하려고 test schema 만듬요
database_name = 'woorifisa'
port = '3306'


logger = logging.getLogger()
logger.setLevel(logging.INFO)
try:
  db_url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}'
  engine = create_engine(db_url)
  # connection = pymysql.connect(host= host, user= username, passwd= password, db= database_name, connect_timeout=5)
  connection = engine.connect()
except pymysql.MySQLError as e:
  logger.error('연결실패!')
  logger.error(e)
  sys.exit()



now = datetime.datetime.now()
previous = now - datetime.timedelta(days=1)
start = previous.strftime("%Y-%m-%d")
end = now.strftime("%Y-%m-%d")
symbol = ['^IXIC','^FTSE','^N225','^STOXX50E','^KS11','^BVSP','^TWII','^BSESN', 'GC=F']
df_nasdaq = yf.download(symbol[0], start, end)
df_ftse = yf.download(symbol[1], start, end)
df_nikkei = yf.download(symbol[2], start, end)
df_euro = yf.download(symbol[3], start, end)
df_kospi = yf.download(symbol[4], start, end)
df_brazil = yf.download(symbol[5], start, end)
df_taiwan = yf.download(symbol[6], start, end)
df_india = yf.download(symbol[7], start, end)
df_gold = yf.download(symbol[8], start, end)


df_nasdaq = df_nasdaq.reset_index()
df_ftse = df_ftse.reset_index()
df_nikkei = df_nikkei.reset_index()
df_euro = df_euro.reset_index()
df_kospi = df_kospi.reset_index()
df_brazil = df_brazil.reset_index()
df_taiwan = df_taiwan.reset_index()
df_india = df_india.reset_index()
df_gold = df_gold.reset_index()


df_nasdaq = df_nasdaq.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_nasdaq['date'] = pd.to_datetime(df_nasdaq['date'], format='%Y-%m-%d')
df_ftse = df_ftse.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_ftse['date'] = pd.to_datetime(df_ftse['date'], format='%Y-%m-%d')
df_nikkei = df_nikkei.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_nikkei['date'] = pd.to_datetime(df_nikkei['date'], format='%Y-%m-%d')
df_euro = df_euro.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_euro['date'] = pd.to_datetime(df_euro['date'], format='%Y-%m-%d')
df_kospi = df_kospi.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_kospi['date'] = pd.to_datetime(df_kospi['date'], format='%Y-%m-%d')
df_brazil = df_brazil.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_brazil['date'] = pd.to_datetime(df_brazil['date'], format='%Y-%m-%d')
df_taiwan = df_taiwan.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_taiwan['date'] = pd.to_datetime(df_taiwan['date'], format='%Y-%m-%d')
df_india = df_india.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_india['date'] = pd.to_datetime(df_india['date'], format='%Y-%m-%d')
df_gold = df_gold.rename(columns={'Date': 'date','Open': 'open','High': 'high','Low':'low','Close': 'close','Adj Close': 'adjclose','Volume': 'volume'})
df_gold['date'] = pd.to_datetime(df_gold['date'], format='%Y-%m-%d')



df_nasdaq['date'] = df_nasdaq['date'].dt.strftime('%Y-%m-%d')
df_ftse['date'] = df_ftse['date'].dt.strftime('%Y-%m-%d')
df_nikkei['date'] = df_nikkei['date'].dt.strftime('%Y-%m-%d')
df_euro['date'] = df_euro['date'].dt.strftime('%Y-%m-%d')
df_kospi['date'] = df_kospi['date'].dt.strftime('%Y-%m-%d')
df_brazil['date'] = df_brazil['date'].dt.strftime('%Y-%m-%d')
df_taiwan['date'] = df_taiwan['date'].dt.strftime('%Y-%m-%d')
df_india['date'] = df_india['date'].dt.strftime('%Y-%m-%d')
df_gold['date'] = df_gold['date'].dt.strftime('%Y-%m-%d')



# 테이블 이름 설정
table_names = ['nasdaq', 'ftse', 'nikkei', 'euro', 'kospi','india', 'taiwan', 'brazil',  'gold']

# DataFrame을 MySQL 테이블에 저장
for i, df in enumerate([df_nasdaq, df_ftse, df_nikkei, df_euro, df_kospi, df_brazil, df_taiwan, df_india, df_gold]):
    table_name = table_names[i]
    
    # DataFrame을 MySQL 테이블로 저장
    df.to_sql(name=table_name, con=connection, if_exists='append', index=False)

    print(f"데이터 저장 완료: {table_name}")

table_names = [ 'kor3y', 'kor10y', 'us3y', 'us10y']

# new_df를 불러옵니다.
new_df = pd.read_csv('bond.csv')

# 'date' 컬럼을 날짜 형식으로 변환하여 'yy-mm-dd' 형식으로 포맷팅합니다.
new_df['date'] = pd.to_datetime(new_df['date']).dt.strftime('%Y-%m-%d')

for col_name, table_name in zip(['kor3y', 'kor10y', 'us3y', 'us10y'], table_names):
    # date 컬럼을 날짜 형식으로 변환
    new_df['date'] = pd.to_datetime(new_df['date']).dt.strftime('%Y-%m-%d')
    
    # 컬럼명에 해당하는 데이터만 선택
    df = new_df[['date', col_name]]
    df.rename(columns={col_name: 'close'}, inplace=True)
    df['open'] = 0
    df['low'] = 0
    df['high'] = 0
    
    # DataFrame을 MySQL 테이블로 저장
    df.to_sql(name=table_name, con=connection, if_exists='replace', index=False)

    print(f"데이터 저장 완료: {table_name}")

connection.commit()