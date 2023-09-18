from flask import Flask
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from everyday import saveTodayPortfolio, getClose

app = Flask(__name__)




# 매일 N시에 실행시킬거다
background_scheduler = BackgroundScheduler(timezone='Asia/Seoul', daemon =True)
background_scheduler.add_job(saveTodayPortfolio,
                             'cron', hour='8', minute='00', id='usingAPI')




if __name__ == "__main__":
    app.run(host="localhost", port=5000)
