import pandas as pd

from CustomModel import runCustom
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
database_name = 'test'
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
  today = '2023-08-29'


  b = runBlack()
  c1, c2 = runCustom()

  b = pd.DataFrame([b])
  c1 = pd.DataFrame([c1])
  c2 = pd.DataFrame([c2])
  df = pd.concat([b, c1, c2])
  df['date'] = [today, today, today]

  df.to_sql(name='portfolio', con=connection, index=True, if_exists='append')
  return 0

def getClose():
  ##FOR PROJECT
  # now = datetime.datetime.now()
  # today = now.strftime("%Y-%m-%d")
  # tomorrow = now + datetime.timedelta(days=1)
  # tomorrow = tomorrow.strftime("%Y-%m-%d")
  ##FOR LOCAL
  today = '2023-08-29'
  tomorrow ='2023-08-30'

  symbol = ['^IXIC', '^FTSE', '^N225', '^STOXX50E', '^KS11', '^BVSP', '^TWII', '^BSESN', 'GC=F']

  df_nikkei = yf.download(symbol[2], today, tomorrow)
  df_kospi = yf.download(symbol[4], today, tomorrow)
  df_taiwan = yf.download(symbol[6], today, tomorrow)
  df_india = yf.download(symbol[7], today, tomorrow)
  dataframes = {
    'df_nikkei' : df_nikkei,
    'df_kospi':df_kospi,
    'df_taiwan' :df_taiwan,
    'df_india' : df_india
  }
  for table_name, df in dataframes.items():
    df.to_sql(name=table_name, con=connection, index=True, if_exists='append')
    time.sleep(10)
  return 0