
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import scipy.io as scio
import scipy as sp

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.datasets import imdb
from keras.optimizers import SGD
import theano

import numpy
from sklearn.decomposition import PCA,RandomizedPCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt


# In[19]:

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


# In[8]:

X=pca.fit()


# In[9]:

Y.shape


# In[20]:

X = pca.transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=7)


# In[21]:

X_train[0]


# In[28]:

epochs = 400
learning_rate = 0.9
decay_rate = learning_rate / epochs
momentum = 0.9
sequence = Input(shape=(X_train.shape[1],), dtype='int32')  
#embedded = Embedding(max_features, 128, input_length=156)(sequence)
embedded = Embedding(X_train.shape[0], 64, input_length=X_train.shape[1])(sequence)

forwards = LSTM(64)(embedded)
backwards = LSTM(64, go_backwards=True)(embedded)

merged = merge([forwards, backwards], mode='sum', concat_axis=-1)
after_dp = Dropout(0.2)(merged)
output = Dense(Y_test.shape[1], activation='softmax')(after_dp)


# In[6]:
model = Model(input=sequence, output=output)



# In[29]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Train...')
#model.fit(X_train, Y_train,batch_size=32,nb_epoch=100,validation_data=[X_test, Y_test])
model.fit(X_train, Y_train, validation_split = 0.10, nb_epoch=epochs, batch_size=20)


'''scores = model.evaluate(X,Y)
# ideally we have to test with X_test and Y_test
print ('val',model.metrics_names[1], scores[1]*100)

Y_pred=model.predict(X)
Result=[np.argmax(temp) for temp in Y_pred] 
Result_2=[np.argmax(temp) for temp in Y]
Output=[(Result[it],Result_2[it]) for it in range(len(Result))]
print ('Accuracy ' , accuracy_score(Result_2,Result))'''




# In[30]:

scores = model.evaluate(X,Y)
# ideally we have to test with X_test and Y_test
print ('val',model.metrics_names[1], scores[1]*100)

Y_pred=model.predict(X)
Result=[np.argmax(temp) for temp in Y_pred] 
Result_2=[np.argmax(temp) for temp in Y]
Output=[(Result[it],Result_2[it]) for it in range(len(Result))]
print ('Accuracy ' , accuracy_score(Result_2,Result))


# In[31]:

scores = model.evaluate(X_test,Y_test)
# ideally we have to test with X_test and Y_test
print ('val',model.metrics_names[1], scores[1]*100)


# In[32]:

scores = model.evaluate(X_test,Y_test)
# ideally we have to test with X_test and Y_test
print ('val',model.metrics_names[1], scores[1]*100)


# In[35]:

Output


# In[36]:

len(Output)


# In[37]:

0.73*0.33*196


# In[ ]:



