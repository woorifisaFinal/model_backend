from flask import Flask
import schedule

from getClose import getClose

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
  schedule.every().day.at("23:00").do(getClose())
  return 'Hello World!'


if __name__ == '__main__':
  app.run()
