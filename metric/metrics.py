#encoding=utf-8
import sklearn
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss

# np.set_printoptions(threshold='nan')


def accuracy_subset(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    accuracy=accuracy_score(y_true,y_pred)
    return accuracy

def accuracy_mean(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    accuracy=np.mean(np.equal(y_true,y_pred))
    return accuracy

def accuracy_multiclass(y_true,y_pred):
    accuracy=accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
    return accuracy

def fscore(y_true,y_pred,threash=0.5,type='micro'):
    y_pred=np.where(y_pred>threash,1,0)
    return f1_score(y_pred,y_true,average=type)

def hamming_distance(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    return hamming_loss(y_true,y_pred)

def fscore_class(y_true,y_pred,type='micro'):
    return f1_score(np.argmax(y_pred,1),np.argmax(y_true,1),average=type)


