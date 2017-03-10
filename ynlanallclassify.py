# Author: Aparajita Haldar (@ahaldar)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

import numpy
from sklearn.decomposition import PCA,RandomizedPCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Accent):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X = numpy.loadtxt("inputvalrand_all.txt", delimiter=',')
Y = numpy.loadtxt("outputvalrand_all.txt", delimiter=',')
pca = PCA(n_components=15,copy=True,whiten=True)
pca.fit(X)
X = pca.transform(X)

# create model
model = Sequential()
model.add(Dense(5, input_dim=15, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='softmax'))


epochs = 200
learning_rate = 0.9
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=7)
# fit the model
model.fit(X, Y, validation_split = 0.33, nb_epoch=epochs, batch_size=30)

# evaluate the model
scores = model.evaluate(X,Y)
print "\nval:%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

# calculate predictions
Y_predictions=model.predict(X_test)
print '-----------------------------------'
print Y_predictions
predictions = model.predict(X)
Y_pred = [round(x) for i in Y_predictions for x in i]
print Y_pred
# round predictions
#rounded = [round(x) for x in predictions]
#print rounded
yint =[]
tp=1
fn=1
tn=1
fp=1
for i in Y:
	yint.append(i)
print yint
"""
for i in range(len(yint)):
	if yint[i]==rounded[i]:
		if yint[i]==1:
			tp+=1
		else:
			tn+=1
	else:
		if yint[i]==1:
			fn+=1
		else:
			fp+=1
print "true positive: ",tp
print "false positive: ",fp
print "false negative: ",fn
print "true negative: ",tn
"""
conf = confusion_matrix(Y_test, Y_pred,labels=[0,1,2,3])
class_names=[0,1]
print conf
plt.figure()
plot_confusion_matrix(conf, classes=class_names,title='Confusion matrix, without normalization')
plt.show()

