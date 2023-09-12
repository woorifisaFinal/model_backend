import pandas as pd

from CustomModel import custom_model
from AlphaModel import alpha_model
from blackLitterman import runBlack
import datetime
import sys
import time
import pymysql
from sqlalchemy import create_engine
import json
import logging
import yfinance as yf

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

logger.info('연결 성공')
def saveTodayPortfolio():
  ##FOR PROJECT
  # now = datetime.datetime.now()
  # today = now.strftime("%Y-%m-%d")
  ##FOR LOCAL
  today = '2022-08-01'


  a1, a2 = alpha_model()


  b1, b2 = runBlack()
  b1_ = json.loads(b1)
  b2_ = json.loads(b2)

  c1, c2 = custom_model(today)


  a1 = pd.DataFrame(a1).T
  a1 = a1.astype('float')
  a1['type'] = "A/안정형"
  a1['type'] = a1['type'].astype(str)
    
  a2 = pd.DataFrame(a2).T
  a2 = a2.astype('float')
  a2['type'] = 'A/공격형'
  a2['type'] = a2['type'].astype(str)

  b1 = pd.json_normalize(b1_)
  b1 = b1.astype('float')
  b1['type'] = 'B/공격형'
  b1['type'] = b1['type'].astype(str)

  b2 = pd.json_normalize(b2_)
  b2 = b2.astype('float')
  b2['type'] = 'B/안정형'
  b2['type'] = b2['type'].astype(str)

  c1 = c1.astype('float')
  c1['type'] = "C/안정형"
  c1['type'] = c1['type'].astype(str)

  c2 = c2.astype('float')
  c2['type'] = 'C/공격형'
  c2['type'] = c2['type'].astype(str)


  df = pd.concat([b1, b2, a1, a2, c1, c2])
  df['date'] = today
  df.to_sql(name='portfolio', con=connection, index=False, if_exists='append',  schema='woorifisa')
  connection.commit()
  return 0

saveTodayPortfolio()



def getClose():
  ##FOR PROJECT
  # now = datetime.datetime.now()
  # today = now.strftime("%Y-%m-%d")
  # tomorrow = now + datetime.timedelta(days=1)
  # tomorrow = tomorrow.strftime("%Y-%m-%d")
  ##FOR LOCAL
  start='2017-01-01'
  end='2023-08-31' 
  symbol = ['^IXIC','^FTSE','^N225','^STOXX50E','^KS11','^BVSP','^TWII','^BSESN', 'GC=F']
  col_list = ['us', 'uk', 'jp', 'euro', 'kor', 'ind', 'tw', 'br', 'gold'] 
  df = yf.download(symbol, start, end)
  dfs = df['Close']
  rename_dict = {old_col: new_col for old_col, new_col in zip(dfs.columns, col_list)}
  dfs = dfs.rename(columns=rename_dict)

  dfs.to_sql(name='symboltest', con=connection, index=True, if_exists='append',  schema='woorifisa')
  connection.commit()
  return 0

# getClose()