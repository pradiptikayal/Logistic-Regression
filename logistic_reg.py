import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

data=np.loadtxt('ex2data1.txt', delimiter=',')
def splittesttrain(data):
	X = data[:, 0:2]
	Y = data[:, 2]
	X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)
	X_train.shape
	return X_train, X_test, Y_train, Y_test

def applyLR(X_train,Y_train):
	classifier=LogisticRegression(random_state=0)
	classifier.fit(X_train,Y_train)
	return classifier
		
def predict(X_test, Y_test, classifier):
	y_pred=classifier.predict(X_test)
	from sklearn.metrics import confusion_matrix
	confusion_matrix=confusion_matrix(Y_test,y_pred)
	print "Confusion matrix:\n"
	print confusion_matrix

def testAccuracy(X_test,Y_test,classifier):
	print "Accuracy:{:.2f}".format(classifier.score(X_test,Y_test))
	
X_train,X_test,Y_train,Y_test=splittesttrain(data)
classifier=applyLR(X_train,Y_train)
predict(X_test,Y_test,classifier)
testAccuracy(X_test,Y_test,classifier)
	
