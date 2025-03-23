# -*- coding: utf-8 -*-
"""

@author: Yassine
"""

#############################################LIBRAIRIES###############################################
import numpy as np
import cv2
import mahotas
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn import metrics #Generate accuracy score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle


#############################################PARAMETRES###############################################
bins = 8
X_TrainD = []
X_TestD = []
seed = 9
num_trees = 100

#############################################UNPICKLING FASE##########################################
#Function to unpickle data:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Unpickling training data
for i in range(1,6):
    globals()['dict{}'.format(i)] = eval("unpickle('cifar-10-batches-py/data_batch_{}')".format(i))

#Unpickling test data
test = unpickle("cifar-10-batches-py/test_batch")

############################################GENERATING X VALUES#######################################
#FOR TRAINING
for i in range(1,6):
        globals()['data{}'.format(i)] = eval("dict{}[b'data']".format(i))

X_Train = np.concatenate((data1, data2, data3, data4, data5))

#FOR TESTING
X_Test = test[b'data']

############################################GENERATING Y VALUES#######################################
#FOR TRAINING
for i in range(1,6):
        globals()['label{}'.format(i)] = eval("dict{}[b'labels']".format(i))

Y_Train = label1 + label2 + label3 + label4 + label5

#FOR TESTING
Y_Test = test[b'labels']

#AS ARRAYS
Y_Train = np.array(Y_Train)
Y_Test = np.array(Y_Test)
############################################DESCRIPTORS###############################################
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

#APPLYING DESCRIPTORS ON TRAIN DATA
for row in X_Train:
    row = row.reshape(32,32,3)
    
    fv_hu_moments = fd_hu_moments(row)
    fv_haralick   = fd_haralick(row)
    fv_histogram  = fd_histogram(row)
    
    StackTrain = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    X_TrainD.append(StackTrain)
    
#APPLYING DESCRIPTORS ON TEST DATA
for row in X_Test:
    row = row.reshape(32,32,3)
    
    fv_hu_moments = fd_hu_moments(row)
    fv_haralick   = fd_haralick(row)
    fv_histogram  = fd_histogram(row)
    
    StackTest = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    X_TestD.append(StackTest)
    


######################################NORMALISING DATA####################################################
rescaled_features1 = StandardScaler().fit_transform(X_TrainD)
rescaled_features2 = StandardScaler().fit_transform(X_TestD)

X_TrainD=np.array(rescaled_features1)
X_TestD=np.array(rescaled_features2)


##########################ALGORITHMS INITIALISATION##################################################################
#Define the base models
level0 = list()
level0.append(('KNN',KNeighborsClassifier(n_neighbors=10)))
level0.append(('RF',RandomForestClassifier(n_estimators = 1000)))
level0.append(('SVM',SVC()))
level0.append(('LDA',LinearDiscriminantAnalysis()))

#Define meta learner model
level1 = LogisticRegression(max_iter=2000)

#Define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1)

##########################APPRENTISSAGE##################################################################

#Pour StackingClassifier
model.fit(X_TrainD, Y_Train)
pred = model.predict(X_TestD)
score = metrics.accuracy_score(Y_Test,pred)

#Pour KNN
model1 = KNeighborsClassifier(n_neighbors=10)
model1.fit(X_TrainD, Y_Train)
pred1 = model1.predict(X_TestD)
ScoreKNN = metrics.accuracy_score(Y_Test,pred1)

#Pour SVM
model3 = SVC()
model3.fit(X_TrainD, Y_Train)
pred3 = model3.predict(X_TestD)
ScoreSVM = metrics.accuracy_score(Y_Test,pred3)

#Pour RandomForest
model6 = RandomForestClassifier(n_estimators = 1000)
model6.fit(X_TrainD, Y_Train)
pred6 = model6.predict(X_TestD)
ScoreRF = metrics.accuracy_score(Y_Test,pred6)

#Pour LDA
model7 = LinearDiscriminantAnalysis()
model7.fit(X_TrainD, Y_Train)
pred7 = model7.predict(X_TestD)
ScoreLDA = metrics.accuracy_score(Y_Test,pred7)

############################################RESULTS#############################################################
results=[]
results.append(ScoreKNN)
results.append(ScoreSVM)
results.append(ScoreRF)
results.append(ScoreLDA)
results.append(score)

names=[]
names.append('KNN')
names.append('SVM')
names.append('RF')
names.append('LDA')
names.append('Stacking')

#Affichage du boxplot de comparaison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


#Sauvegarde du model de StackinClassifier
filename = 'ModelML.sav'
pickle.dump(model, open(filename, 'wb'))



