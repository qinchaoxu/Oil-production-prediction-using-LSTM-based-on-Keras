import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

np.random.seed(7)


def load_data(time_step, split):
    
    f = open('oil_data.csv')
    df = pd.read_csv(f)         #读入数据
    data = np.array(df['oilrate']).astype(float)  #获取时间序列
    data_mean = np.mean(data)
    data_max =  max(data) 
    data_min = min(data)
    data_all = (data - np.mean(data)) / (max(data) - min(data)) #np.std(data)#标准化

    data_0 = []
    data_1 = []
    for i in range(len(data_all) - time_step - 1):
        data_0.append(data_all[i: i + time_step + 1])
        data_1.append(data[i: i + time_step + 1])
    reshaped_data = np.array(data_0).astype('float64')
    # np.random.shuffle(reshaped_data)

    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, data_mean, data_max, data_min


def build_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=20, return_sequences=True))
    print(model.layers)
    model.add(LSTM(20, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model





if __name__ == '__main__':
    time_step = 20
    split = 0.2
    train_x, train_y, test_x, test_y, data_mean, data_max, data_min = load_data(time_step=time_step,split=split)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    
    model = build_model()
    model.fit(train_x, train_y, batch_size=40, nb_epoch=40, validation_split=0.1)

    predict_train = model.predict(train_x)

    pre_seq = test_x[0]
    pre_seq = np.reshape(pre_seq, (1, time_step, 1))
    # print(pre_seq)
    predict_y = []

    next_seq = model.predict(pre_seq)
    # print(next_seq)
    for i in range(500):
        next_seq = model.predict(pre_seq)
        # print(next_seq)
        predict_y.append(list(next_seq[0]))
        pre_seq = np.vstack((pre_seq[0][1:], next_seq[0]))
        pre_seq = np.reshape(pre_seq, (1, time_step, 1))

    

    print(predict_y)
    dataframe = pd.DataFrame(predict_train)
    # dataframe.to_csv('out_train_1.csv') #保存
    dataframe = pd.DataFrame(predict_y)
    # dataframe.to_csv('out_y_1.csv') #保存


    fig1 = plt.figure(1)
    plt.plot(predict_train, 'g:')
    plt.plot(train_y, 'r-')
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(predict_y, 'g:')
    plt.plot(test_y, 'r-')
    plt.show()

    print(data_mean, data_max, data_min)