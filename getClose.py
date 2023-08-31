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




# getClose()