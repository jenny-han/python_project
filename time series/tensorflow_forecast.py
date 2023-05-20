#%%
# RNN(Recurrent Neural Networks) 과거 주가 데이터 학습하여 다음날 종가 예측하기
import tensorflow as tf # rnn
import numpy as np 
import matplotlib #graph
import os
import matplotlib.pyplot as plt #graph

# 모든 연산에 의해 생성된 난수 시퀀스들이 세션간 반복이 가능하게 하기위해서, 
# 그래프 수준의 시드를 설정합니다.
# -> 랜덤에 의해 똑같은 결과를 재현하도록 시드설정.
# 하이퍼 파라미터를 튜닝하기 위한 용도 
# (흔들릴경우 무엇때문에 좋아졌는지 파악하기 어려움 )
tf.set_random_seed(777)

#함수를 정의합니다.
def MinMaxScaler(data) :
    #데이터의 모든 숫자들을 최소값만큼 뺀다.
    numerator = data - np.min(data,0)
    #최대값과 최소값의 차이를 구한다.
    denominator = np.max(data,0)-np.min(data,0)
    #너무 큰 값이 나오지 않도록 나눠준다.
    return numerator/(denominator+1e-7)

# 하이퍼파라미터
    #신경망 학습을 통해서 튜닝 또는 최적화 해야하는 주변수가 아니라,
    # 학습 진도율이나 일반화 변수처럼, 사람들이 선험적 지식으로 설정을 하거나 
    # 또는 외부 모델 메커니즘을 통해 자동으로 설정이 되는 변수
seq_length= 7    #1개 시쿼스의 길이(시계열 데이터 입력 개수)
data_dim = 5     #Variable개수
hidden_dim = 10 #각 셀의 출력크기
output_dim =1   #결과분류 총 개수
learing_rate =0.01  #학습률
epoch_num=500   #에폭 횟수 (학습용 전체데이터를 몇 회 반복하여 학습할 것인지 입력)

#데이터를 로딩한다.
# 시작가 고가 저가 거래량 종가
xy = np.loadtxt('data/data-02-stock_daily.csv',delimiter=',')
#%%
#데이터 전처리
xy = xy[::-1] #제일 앞을 뒤로, 제일뒤를 앞으로 순서(index)를 뒤집는다.
print("xy[0][0] : ", xy[0][0]) #568.000257
#%%
xy =MinMaxScaler(xy)
print("xy[0][0] : ",xy[0][0]) #0.21375105364038344
#%%
x = xy
y= xy[:,[-1]] #마지막열이 주식종가(정답) 이다.
print("x[0] : " ,x[0])
#x[0] :  [0.21375105 0.20817981 0.19179183 0.00046608 0.1920924 ]
print("y[0] : " ,y[0] )#y[0] :  [0.1920924]
#%%
dataX = []
dataY = []
for i in range(0,len(y)-seq_length) :
    _x = x[i : i+seq_length]
    _y = y[i+seq_length] #다음날 나타날 주가 (정답)
    if i is 0 : 
        print(_x,"->",_y)
        dataX.append(_x)
        dataY.append(_y)
#%%
#학습용/테스트용 데이터 생성
#70%를 테스트용 데이터로 사용
train_size = int(len(dataY)* 0.7)
#나머지30%를 테스트용 데이터로 사용
train_size = len(dataY) - train_size

#데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])


#데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])
#%%
#텐서플로우 플레이스홀더 생성
#tf.placeholder(dtype:데이터형,shape:행렬의 차원,name:이름정의) 
#학습용 , 테스트용으로 X,Y 를 생성
X = tf.placeholder(tf.float32,[None,seq_length,data_dim])
print("X: ",X)
Y = tf.placeholder(tf.float32,[None,1])
print("Y: ",Y)

#검증용 측정지표를 산출하기위한 targets, predictions 생성
targets = tf.placeholder(tf.float32,[None,1])
print("targets : ",targets)
predictions = tf.placeholder(tf.float32,[None,1])
print("predictions : ",predictions)
#%%
#모델(LSTM네트워크) 생성
def lstm_cell() :
    #LSTM셀을 생성한다.
    #num_units : 각 셀의 출력크기
    #forget_bias : 편향 added to forget gates
    #state_is_tuple : True --> 2개의상태(c_state,m_state)가 반환되고 접근된다.
    #state_is_tuple : False --> 그들은 열의 축으로 연결된다
    #cell = tf.contrib.rnn.BasicLSTMCell
(num_units=hidden_dim,state_is_tuple=True,activation=tf.sigmoid)
    #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=True,
activation=tf.tanh)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,forget_bias=0.8,
                                        state_is_tuple=True,activation=tf.tanh)
    #tanh(hyperbolic tangent) function은 
    #sigmoid 처럼 비선형 함수이지만 결과값의 범위가 -1부터 1이기 때문에 
    #sigmoid와 달리 중심값이 0입니다. 
    #따라서 sigmoid보다 optimazation이 빠르다는 장점이 있고, 항상 선호됩니다. 
    #하지만 여전히 vanishing gradient(사라지는 증감률/변화) 문제가 발생하기 때문에 대안이 등장
    return cell
# 몇개의 층으로 쌓인 Stacked RNNs 생성, - 여기에서는 1개층만!
multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(1)],state_is_tuple=True)
#%% 재시도 시 기존에 있다고 해서 지워주는 코드 넣어줌
#RNN Cell (여기에서는 LSTM셀이다) 들을 연결
train_graph = tf.Graph()
with tf.Graph().as_default():
    hypothesis,_states = tf.nn.dynamic_rnn(multi_cells,X,dtype=tf.float32)
    print("hypothesis :",hypothesis)
#%%
Y_pred = tf.contrib.layers.fully_connected(hypothesis[:,-1],output_dim,activation_fn=None)
#%%
loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)

#RMSE(Root Mean Square Error)
#rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))
#%%
with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #학습한다
    for epoch in range(epoch_num) : 
        _,step_loss = sess.run([train,loss],feed_dict={X:trX,Y:trY})
        print("[step:{} loss :{}]".format(epoch,step_loss))
        
    #테스트한다.
    test_predict = sess.run(Y_pred,feed_dict={X:teX})
    
    #테스트용 데이터를 기준으로 측정지표 rmse를 산출
    rmse_val = sess.run(rmse,feed_dict={targets:teY,predictions:test_predict})
    print("rmse : ",rmse_val)
    #%% 그래프를 생성합니다.
    plt.plot(testY,'r')
    plt.plot(test_predict,'b')
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

