from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
from sklearn import neighbors
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import time
from sklearn.svm import SVC

scaler = MinMaxScaler()

#load Data
TrainMnist = np.loadtxt('mnist_train.csv',delimiter=",", dtype=np.float32, skiprows=1)
TestMnist = np.loadtxt('mnist_test.csv',delimiter=",", dtype=np.float32, skiprows=1)


labels = TrainMnist[:,0:1]
TrainMnist = TrainMnist[:,1:]

labelst = TestMnist[:,0:1]
TestMnist = TestMnist[:,1:]

for i in range(60000):
    if (labels[i] % 2 == 0):
        labels[i] = 0  # Evens
    else:
        labels[i] = 1  # Odds

for i in range(10000):
    if (labelst[i] % 2 == 0):
        labelst[i] = 0  # Evens
    else:
        labelst[i] = 1  # Odds

y_train=np.ravel(labels)
x_train=np.array(TrainMnist)

y_test=np.ravel(labelst)
x_test=np.array(TestMnist)

#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

main_pca = PCA(n_components=90)
x_train = main_pca.fit_transform(x_train)

main_pca = PCA(n_components=90)
x_test = main_pca.fit_transform(x_test)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#time1=time.time()

#svm = LinearSVC(loss='hinge', multi_class='ovr', C=1)
#svm.fit(x_train, y_train)
#print("Train, test acc:", 100*svm.score(x_train, y_train), 100*svm.score(x_test, y_test))

#time2=time.time()
#print('Time Passed: {:.3f}sec'.format(time2-time1))

#C=10
#time1=time.time()
#svm = LinearSVC(loss='hinge',C=C, multi_class='ovr', max_iter=1000000)
#svm.fit(x_train, y_train)
#print("C = ", C, "Train, test acc: ", 100*svm.score(x_train, y_train), 100*svm.score(x_test, y_test))
#time2=time.time()
#print('Time Passed: {:.3f}sec'.format(time2-time1))
    

C=1
time1=time.time()
svm = SVC(C=C, kernel='linear', decision_function_shape='ovr', max_iter=100000)
svm.fit(x_train, y_train)
print("C = ", C, "Train, test acc: ", 100*svm.score(x_train, y_train), 100*svm.score(x_test, y_test))
time2=time.time()
print('Time Passed: {:.3f}sec'.format(time2-time1))


#C=10
#time1=time.time()
#svm = SVC(C=C, kernel='poly', degree=5, decision_function_shape='ovr', max_iter=100000)
#svm.fit(x_train, y_train)
#print("C = ", C, "Train, test acc: ", 100*svm.score(x_train, y_train), 100*svm.score(x_test, y_test))
#time2=time.time()
#print('Time Passed: {:.3f}sec'.format(time2-time1))

#gamma=10
#time1=time.time()
#svm = SVC(C=1, kernel='rbf', gamma=gamma, decision_function_shape='ovr', max_iter=100000)
#svm.fit(x_train, y_train)
#print("gamma = ", gamma, "Train, test acc: ", 100*svm.score(x_train, y_train), 100*svm.score(x_test, y_test))
#time2=time.time()
#print('Time Passed: {:.3f}sec'.format(time2-time1))
    
