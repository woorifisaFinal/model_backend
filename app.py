from flask import Flask, request, jsonify
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from everyday import saveTodayPortfolio, getClose

app = Flask(__name__)




# 매일 N시에 실행시킬거다
background_scheduler = BackgroundScheduler(timezone='Asia/Seoul', daemon =True)
background_scheduler.add_job(saveTodayPortfolio,
                             'cron', hour='8', minute='00', id='saveTodayPortfolio')
background_scheduler.add_job(getClose,
                             'cron', hour='23', minute='00', id='getClose')

@app.route('/')
def hello_world():  # put application's code here
  return 'Hello World!'

# POST 요청을 처리하는 엔드포인트
@app.route('/portfolio/infer', methods=['POST'])
def infer():


    # 스프링에서 넘어온 json 데이터를 변수에 저장합니다.
    dto_json = request.get_json() 
    #json데이터를 numpy로 변환(인코딩)
    


    # 인퍼런스 과정 추가


    # 인퍼런스의 결과를 (key) 바꾸는 작업 

    # 예시로 기대하는 응답 데이터를 생성하여 반환
    response_data = {
        "type": dto_json["type"],
        "datetime": "2022-08-08",
        "KS": 0,
        "NA": 0,
        "EU": 0,
        "BZ": 0,
        "TW": 0,
        "IN": 0,
        "JP": 0,
        "UK": 0,
        "GOLD": 0,
        "KRBondLong": 0,
        "KRBondShort": 0
    }
    # model result -> jinja2
    # model.py -> import 해서 함수 사용

    # response = Response()
    # response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
