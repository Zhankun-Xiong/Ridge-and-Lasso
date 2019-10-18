import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
trainratio=0.7         #training data ratio


def get_data():
    alldata=np.zeros((506,14))
    data=np.loadtxt('housing.csv',dtype='str',delimiter=',')
    for i in range(data.shape[0]):
        for x in range(14):
            alldata[i][x]=float(data[i][1:].split(' ')[x])

    return alldata

data=get_data()
trainnum=int(data.shape[0]*trainratio)
x=np.arange(506)
plt.xlabel("all sample")
plt.ylabel("value")
plt.scatter(x,data[:,13])
plt.show()
plt.clf()
#data split
trainX=data[0:trainnum,0:13]
trainY=data[0:trainnum,13]
testX=data[trainnum:,0:13]
testY=data[trainnum:,13]
print(trainX.shape)
print(testX.shape)

alpha1=[0.1,0.5,1]
color=['blue','green','gold']
#model training
for i in range(1):
    from sklearn.linear_model import Ridge #（91 xianxing）
    ridge=Ridge(alpha=0.1)
    ridge.fit(trainX,trainY)
    #result Analysis

    # w result and paint and save picture
    coef_multi_task_ridge_ = ridge.coef_
    x=np.arange(13)
    plt.xlabel("w")
    plt.ylabel("value")
    plt.scatter(x, coef_multi_task_ridge_, color=color[i])
    plt.plot(x,coef_multi_task_ridge_,color=color[i])
    plt.savefig('ridge_w_result.png')
    plt.clf()

    # training regression result
    x = np.arange(354)
    plt.xlabel("train_sample")
    plt.ylabel("price")
    plt.scatter(x, trainY, marker='x')
    plt.plot(x, ridge.predict(trainX), c='r')
    plt.savefig('ridge_regression_train_result.png')
    plt.clf()

    # regression result
    x=np.arange(152)
    plt.xlabel("test_sample")
    plt.ylabel("price")
    plt.scatter(x, testY, marker='x')
    plt.plot(x, ridge.predict(testX),c='r')
    plt.savefig('ridge_regression_result.png')

    # training set regression condition
    print('平均绝对值误差：', sm.mean_absolute_error(trainY, ridge.predict(trainX)))
    print('平均平方误差：', sm.mean_squared_error(trainY, ridge.predict(trainX)))
    print('中位绝对值误差：', sm.median_absolute_error(trainY, ridge.predict(trainX)))


    # testing set regression condition
    print('平均绝对值误差：', sm.mean_absolute_error(testY, ridge.predict(testX)))
    print('平均平方误差：', sm.mean_squared_error(testY, ridge.predict(testX)))
    print('中位绝对值误差：', sm.median_absolute_error(testY, ridge.predict(testX)))






    
    
