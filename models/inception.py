import scipy
import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix,coo_matrix





def getData(dataname,linktype,ispart=False):
    data=sio.loadmat('data/hetedataset/'+dataname+'.mat')

    truelabels=data['label']
    truefeatures = data['feature']
    rownetworks = data[linktype][0]
    if ispart:
        knownindex = data['knownindex'][0]
    else:
        knownindex=None
    return truelabels,truefeatures,rownetworks,knownindex

def sparse_to_tuple(matrix):
    if not scipy.sparse.isspmatrix_coo(matrix):
        matrix=matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape
