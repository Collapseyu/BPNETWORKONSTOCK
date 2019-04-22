import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
from matplotlib import pyplot as plt
import wave_inference
BATCH_SIZE=50 #guiyihua 500 label[12]
LEARNING_RATE_BASE=0.001 #guiyihua 0.01 label[12]
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINING_STEPS=50000
MOVING_AVERAGE_DECAY=0.99
TRAINNUM=2500
MODEL_SAVE_PATH="./"
MODE_NAME="model_base.ckpt"
def noOne(trainData,testData):
    newTrain=[]
    newTest=[]
    for i in trainData:
        tmp=[]
        for j in range(12):
            tmp.append(float(i[j]))
        tmp.append(float(i[13])/10000)
        tmp.append(float(i[14])/10000)
        tmp.append(float(i[15])/10000)
        tmp.append(float(i[12]))
        tmp.append(float(i[16]))
        tmp.append(float(i[17]))
        newTrain.append(tmp)
    for i in testData:
        tmp = []
        for j in range(12):
            tmp.append(float(i[j]))
        tmp.append(float(i[13]) / 10000)
        tmp.append(float(i[14]) / 10000)
        tmp.append(float(i[15]) / 10000)
        tmp.append(float(i[12]))
        tmp.append(float(i[16]))
        tmp.append(float(i[17]))
        newTest.append(tmp)

    trainData=np.array(newTrain,dtype=np.float32)
    testData=np.array(newTest,dtype=np.float32)
    normXData=trainData[: , :15]
    tmpData = []
    for i in trainData:
        tmpData.append([i[15]])
    normYData = np.array(tmpData)
    normXtestData = testData[:, :15]
    tmpData = []
    for i in testData:
        tmpData.append([i[15]])
    normYtestData = np.array(tmpData)
    return normXData,normYData,normXtestData,normYtestData

def train(normXData,normYData,normXtestData,normYtestData):#(xData,yData,xTestData,yTestData):

    x=tf.placeholder(tf.float32,[None,wave_inference.INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,wave_inference.OUTPUT_NODE],name='y-input')

    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=wave_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    #cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #cross_entropy_mean=tf.reduce_mean(cross_entropy)
    mse=tf.reduce_mean(tf.abs(y_-y))#tf.losses.mean_squared_error(labels=y_,predictions=y)#tf.reduce_mean(tf.abs(y_-y)) #归一化后损失函数
    m1=tf.reduce_mean(y_)
    m2=tf.reduce_mean(y)
    mse1=tf.sqrt(tf.reduce_mean(tf.square(y_-y)))
    loss=mse
    #loss=mse+tf.add_n(tf.get_collection('losses'))#cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        TRAINNUM/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step) #learning_rate
    #aa1=scaler.inverse_transform([[0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0,0, m1, 0.0567642, 0]])
    #bb1=scaler.inverse_transform([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,m2, 0.0341829, 0]])
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            start=(i*BATCH_SIZE)%(TRAINNUM-BATCH_SIZE)
            end=start+BATCH_SIZE
            xs=normXData[start:end]
            ys=normYData[start:end]
            #xs,ys=mnist.train.next_batch(BATCH_SIZE)

            _, loss_value, step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys}) #[train_op,loss,global_step]
            if i%1000==0:
                p,l=sess.run([y,y_],feed_dict={x:normXtestData,y_:normYtestData})
                #print("%g %g"%(loss_value_n,b1))
                predictions = np.array(p).squeeze()
                labels = np.array(l).squeeze()
                rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
                print("After %d training step(s),loss: %g" %(step,rmse))
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODE_NAME),global_step=global_step)
        p,l=sess.run([y,y_],feed_dict={x:normXtestData,y_:normYtestData})
        #predictions = np.array(p[0]).squeeze()
        #labels = np.array(l[0]).squeeze()
        rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
        print("rmse=%g"%rmse)
        plt.figure()
        plt.plot(p, label='predictions')
        plt.plot(l, label='real')
        plt.legend()
        plt.show()

def main(argv=None):
    #mnist = input_data.read_data_sets("/path/to/MINST_data/", one_hot=True)
    #train(mnist)
    csv_file=csv.reader(open('bpDataWithoutWave.csv','r'))
    all=[]
    trainSet=[]
    testSet=[]
    for i in csv_file:
        all.append(i)
    count_n=0
    for i in all:
        if float(i[18]) > 10:
            continue
        if count_n<2500: #2500 without
            trainSet.append(i[2:])
        elif count_n<3000:
            testSet.append(i[2:])
        count_n+=1
    """
    for i in range(2500):
        trainSet.append(all[i][2:])
    for i in range(2500,3000):
        testSet.append(all[i][2:])
    """
    #print('a')
    #print(type(normYtestData))
    normXData, normYData, normXtestData, normYtestData=noOne(trainSet,testSet)
    train(normXData,normYData,normXtestData,normYtestData) #0.201672. 0.367379  0.0244033  0.202092  0.36978  0.0441577
    print('a')
if __name__=='__main__':
    tf.app.run()