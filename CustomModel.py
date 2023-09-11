def sharpe_loss(y_true, y_pred):
    """
    y_true : window size만큼의 실제 종가
    """
    # print("\n\ny_true.shape:", y_true.shape)
    # print("y_pred.shape:", y_pred.shape)
    # print("y_true : ", tf.get_static_value(y_true))
    # make all time-series start at 1
    # first_day_per_batch = y_true[0]
    first_day_per_batch = tf.expand_dims(y_true[:,0], axis=1)
    data = tf.divide(y_true, first_day_per_batch)
    # print("data value : ", tf.get_static_value(data))
    y_pred = tf.expand_dims(y_pred, axis=1)


    # print("y_pred value : ", tf.get_static_value(y_pred))
    # value of the portfolio after allocations applied
    portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=2)

    # print("portfolio_values : ", portfolio_values)
    portfolio_returns = (portfolio_values[:, 1:] - portfolio_values[:, :-1]) / portfolio_values[:, :-1]  # % change formula
    # print("portfolio_returns : ", portfolio_returns)
    sharpe = tf.reduce_sum(K.mean(portfolio_returns, axis=1) / K.std(portfolio_returns, axis=1))
    # sharpe = tf.divide(tf.reduce_sum(K.mean(portfolio_returns, axis=1) / K.std(portfolio_returns, axis=1)), tf.constant([100.0]))

    # 배치크기로 나눠주기
    sharpe = tf.divide(sharpe, tf.cast(tf.shape(y_pred), dtype=tf.dtypes.float32)[0])

    # print("sharpe value : ", tf.get_static_value(sharpe))
    # print("sharpe : ", sharpe)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of outputs negated function is its max)
    return -sharpe


def build_model(input_shape, outputs):

    # 논문 방식의 BaseLine
    # model = Sequential([
    #     LSTM(64, input_shape=input_shape),
    #     Flatten(),
    #     Dense(outputs, activation='softmax')
    # ])

    inp = tf.keras.Input(shape=input_shape)

    RNN = getattr(layers, 'LSTM') # 'GRU', LSTM'
    # x = RNN(units=64, return_sequences=False)(inp)
    x = RNN(units=64, return_sequences=True)(inp)
    x = RNN(units=32, return_sequences=True)(x)
    x = RNN(units=8, return_sequences=False)(x)
    x = layers.Dense(outputs,activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # debbuging Tips https://keras.io/examples/keras_recipes/debugging_tips/#tip-3-to-debug-what-happens-during-fit-use-runeagerlytrue
    model.compile(loss=sharpe_loss, optimizer=opt) # run_eagerly=True



    return model


def make_data(df_, lookback_window, lookahead_window):
    df = df_.copy()

    # interleave로 해봐도 될 듯
    # data with returns
    # data_w_ret = np.concatenate([ df.values[1:], df.pct_change().values[1:] ], axis=1)

    # df = df.iloc[1:] #.values

    data_w_ret = np.dstack([df.values[1:], df.pct_change().values[1:]])
    df = df.iloc[1:] #.values
    data_w_ret = data_w_ret.reshape(df.shape[0],-1)

    # total_sample_num = df.shape[0]-window_size+1
    total_sample_num = df.shape[0]-(lookback_window+lookahead_window)+1 # lookback을 위한, lookahead를 위한 두 개 빼주기

    x_data = np.zeros((total_sample_num, lookback_window, data_w_ret.shape[1]))
    # unsupervised 방식이지만 sharpe에서 활용하도록 작성
    y_data = np.zeros((total_sample_num, lookahead_window, df.shape[1]))

    portfolio_date = []
    # for idx in tqdm(range(total_sample_num)):
    # standard_day를 기준으로 -lookback_window만큼 뒤를 보고, lookahead_window만큼 앞을 보기 위해 for문 조정
    # 예전 idx를 standard_day로 바꾼 것
    for idx, standard_day in tqdm(enumerate(range(lookback_window, total_sample_num+lookback_window)), total=total_sample_num):

        # x_data[idx,] = data_w_ret[idx:idx+window_size]
        x_data[idx,] = data_w_ret[standard_day-lookback_window:standard_day]

        y_data[idx,] = df[standard_day:standard_day+lookahead_window]

        portfolio_date.append(df.index[standard_day])

        # portfoilo_day.append

    return x_data, y_data, portfolio_date

# from tensorflow.keras.callbacks import ModelCheckpoint  # 우리가 history 변수에 임의로 저장해왔던 것처럼
# # 모델이 학습을 하면서 어떤 특이점을 만날 때마다 누적해서 저장해주는 콜백
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 사용자 정의 ReduceLROnPlateau 콜백 클래스
class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.2, patience=2, min_lr=1e-6):
        super(CustomReduceLROnPlateau, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return

        val_loss_abs = np.abs(val_loss)

        if epoch == 0:
            self.best_loss = val_loss_abs
        elif val_loss_abs < self.best_loss:
            self.best_loss = val_loss_abs
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = max(current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'Reducing learning rate to {new_lr} at epoch {epoch + 1}')
                self.wait = 0

# 사용자 정의 콜백 객체 생성
custom_reduce_lr = CustomReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-6)

log_filename2 = '/content/drive/MyDrive/ITStudy/logs/training_log_LR_21_val_class_1e-6_epoch10_final.csv'

csv_logger = CSVLogger(log_filename2)

# Ref : https://github.com/shilewenuw/deep-learning-portfolio-optimization/blob/main/Model.py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import tensorflow.keras.backend as K


pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_rows = None
np.set_printoptions(precision=6, suppress=True)
# setting the seed allows for reproducible results
np.random.seed(123)

#### 각 자산별 Close 받아오기
df = pd.read_csv("/content/drive/MyDrive/ITStudy/파이널프젝/total_17_22.csv",index_col=0)

train = df[df.index<('2021')]
# valid = df[df.index>=('2021')]
valid = df[(df.index>=('2021')) & (df.index<('2022'))]

# valid = pd.read_csv("/content/stage1_prediction_22.csv",index_col=0)
# valid[:] = np.random.randint(100, 2000, size=(valid.shape)).astype(np.float32)


# window_size = 50
lookback_window = 50
lookahead_window = 30
# lookahead_size = 90

# x_data, y_data = make_data(train, window_size)
x_data, y_data, portfolio_train_date = make_data(train, lookback_window, lookahead_window)
# logger.info(f"!!Train data infoi!! \n  x_data.shape : {x_data.shape} \t y_data.shape : {y_data.shape}")


### STANDARIZE
# x_data,y_data = scaler(x_data,y_data, cfg.base, is_train=True)

# X_train, X_valid, y_train, y_valid = train_test_split(x_data, y_data, shuffle=True,random_state=cfg.base.seed, test_size=0.2)
x_valid, y_valid, portfolio_valid_date = make_data(valid, lookback_window, lookahead_window)

### 모델
model = build_model(x_data[0].shape, y_data.shape[2])

### 학습
history = model.fit(x_data,y_data,
                validation_data = (x_valid,y_valid),
                # sample_weight = np.tile(w,GRP),
                batch_size=4, epochs=10, verbose="auto", callbacks = [csv_logger,custom_reduce_lr])


# 최대 -sharpe 값과 그 때의 y_pred 반환
# max_sharpe, max_sharpe_y_pred = get_max_sharpe_and_y_pred(model, x_valid, y_valid)
#2021 validation val_loss -0.1306

x_valid, y_valid, portfolio_valid_date = make_data(valid, 50,30)
predictions = model.predict(x_valid)
### 모델 저장
# model.save_weights(f"{cfg.base.task_name}_{cfg.base.model_name}_{cfg.base.exp_name}.h5")
model.save_weights("model_w.h5")
# model.load_weights(INFER_FROM_PATH + f'GRU_f{fold}_v{VER}.h5')


predictions_df = pd.DataFrame(predictions)

rename_dict = {
    0:"kor",
    1:"us",
    2:"euro",
    3:"uk",
    4:"jp",
    5:"kor3y",
    6:"kor10y",
    7:"us3y",
    8:"us10y",
    9:"gold",
    10:"br",
    11:"tw",
    12:"ind"
    }

col_list = ['us', 'uk', 'jp', 'euro', 'kor', 'ind', 'tw', 'br', 'kor3y', 'kor10y', 'us3y', 'us10y', 'gold']

predictions_df.rename(columns = rename_dict,inplace = True)

predictions_df = predictions_df[col_list]

portfolio_valid_date_df =pd.DataFrame(portfolio_valid_date, columns = ['date'])

portfolio_valid_date_df['date'] = pd.to_datetime(portfolio_valid_date_df['date']) # 날짜별로 추출하기 위해 date타입으로 변환 
result = pd.concat([predictions_df, portfolio_valid_date_df],axis = 1)

result.set_index(result.columns[-1], inplace=True)

def wish_date_weight(date): # date는 '$$$$-$$-$$'형식으로 받아온다

    stable_asset = ['kor3y','kor10y','us3y','us10y','gold']
    risky_asset = ['us', 'uk' ,'jp',	'euro',	 'ind',	'tw',	'br','kor']

    stable_sum.loc[date] = result.loc[date][stable_asset].sum(axis=0)
    risky_sum.loc[date] = result.loc[date][risky_asset].sum(axis=0)
    

    final = result.copy() #result 원본값을 그대로 보존하기 위해서
    if stable_sum.loc[date]<0.4:
    
        risky_weight.loc[date] = result.loc[date].copy() # 모델의 가중치가 곧 공격형 자산비중
        # print( risky_weight - result)
        final.loc[date][stable_asset] = final.loc[date][stable_asset]*(0.4/stable_sum.loc[date])
    
        final.loc[date][risky_asset] = final.loc[date][risky_asset]*(0.6/risky_sum.loc[date])
       
        stable_weight = final.loc[date] # 수정한 비중으로 안전형 자산

    elif stable_sum.loc[date]>0.4:
    
        stable_weight = result.loc[date].copy() # 모델의 가중치가 곧 안전형 자산 비중
   
        final.loc[date][stable_asset] = final.loc[date][stable_asset]*(0.4/stable_sum.loc[date])
        
        final.loc[date][risky_asset] = final.loc[date][risky_asset]*(0.6/risky_sum.loc[date])
       
        risky_weight = final.loc[date] # 수정한 비중으로 공격형 자산

    return stable_weight, risky_weight

result.to_json("/content/drive/MyDrive/ITStudy/파이널프젝/model_weight.json") #risky_weight, stable_weight 으로 df명 변경해서 출력할 것

