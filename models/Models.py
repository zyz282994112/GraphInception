from keras.engine import merge
from keras.layers import Dense, Dropout, Convolution2D, Activation, Highway
import tensorflow as tf
from keras import backend as K

from models.GraphHighway import GraphHighway


def StackLearning(multinetworks,contentFeature,outputdim,stack_layer,type='sigmoid'):
    relfeature = Dense(outputdim, activation='relu')(contentFeature)
    for i in range(stack_layer+1):
        all_net_layer = []
        for network in multinetworks:
            Net_feature = tf.sparse_tensor_dense_matmul(network, relfeature)
            all_net_layer.append(Net_feature)

        all_feature = merge(all_net_layer+[contentFeature, relfeature], mode='concat', concat_axis=1)
        if i==stack_layer:
            relfeature=Dense(outputdim, activation='relu')(all_feature)
        else:
            y = Dense(outputdim,activation=type)(all_feature)
    return y



def graph_inception_module(inputs,kernelsize,inputdim,outputdim,multinetworks,need1X1=True):
    convresults=[]
    for network in multinetworks:
        Korder=inputs
        for _ in range(kernelsize):
            Korder = tf.sparse_tensor_dense_matmul(network, Korder)
            W=tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
            relfeature=Activation('relu')(K.dot(Korder,W))

            if need1X1:
                relfeature=K.expand_dims(K.expand_dims(relfeature,0),0)

            convresults.append(relfeature)

    convresults = merge(convresults, mode='concat', concat_axis=1)

    if need1X1:
        convresults=Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(convresults)
        convresults=K.reshape(tf.transpose(convresults,perm=[2,0,1,3]),shape=(-1, outputdim))

    return convresults



def GraphInception(multinetworks,contentFeature,relFeature, inputdim,outputdim,layerdepth,kernelsize,hiddendim,outputtype='sigmoid'):

    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        if i==0:
            relFeature=graph_inception_module(relFeature,kernelsize,inputdim,hiddendim,multinetworks,need1x1)
        else:
            relFeature = graph_inception_module(relFeature, kernelsize, hiddendim, hiddendim, multinetworks, need1x1)

    allFeature= merge([contentFeature, relFeature], mode='concat')
    y = Dense(outputdim, activation=outputtype)(allFeature)
    return y



def StackInception(multinetworks,contentFeature,outputdim,layerdepth,kernelsize,hiddendim,outputtype='sigmoid'):
    relFeature = Dense(hiddendim, activation='relu')(contentFeature)

    for i in range(layerdepth):
        if i==layerdepth-1:
            relFeature=graph_inception_module(relFeature, kernelsize, hiddendim, hiddendim, multinetworks, False)
        else:
            relFeature=graph_inception_module(relFeature, kernelsize, hiddendim, hiddendim, multinetworks, True)

    y = Dense(outputdim, activation=outputtype)(relFeature)
    return y



def CLN(multinetworks,contentFeature,hiddendim,outputdim,stack_layer,dropout=0.5,type='sigmoid',ishighway=True,isshared=True):
    relFeature = Dense(hiddendim, activation='relu')(contentFeature)
    relFeature=Dropout(dropout)(relFeature)
    clnlayer=GraphHighway(input_dim=hiddendim, n_rel=len(multinetworks), activation='relu', transform_bias=-0.1*stack_layer)

    def highway(shared):
        if shared: return clnlayer
        return GraphHighway(input_dim=hiddendim, n_rel=len(multinetworks), activation='relu', transform_bias=-0.1*stack_layer)

    for i in range(stack_layer):
        all_net_layer = []
        for network in multinetworks:
            Net_feature = tf.sparse_tensor_dense_matmul(network, relFeature)
            all_net_layer.append(Net_feature)
        netlayer = merge(all_net_layer, mode='concat')
        if ishighway:
            relFeature=highway(isshared)([relFeature,netlayer])
        else:
            relFeature=Dense(hiddendim,activation='relu')(merge([relFeature,netlayer],mode='concat'))
    relFeature = Dropout(dropout)(relFeature)
    y = Dense(outputdim,activation=type)(relFeature)
    return y



def HighwayNet(contentFeature,hiddendim,outputdim,stack_layer,type='sigmoid',isshared=True,dropout=0.5):
    relFeature = Dense(hiddendim, activation='relu')(contentFeature)
    relFeature=Dropout(dropout)(relFeature)
    shareway=Highway(input_dim=hiddendim,activation='relu')

    def highway(shared):
        if shared: return shareway
        return Highway(input_dim=hiddendim,activation='relu')

    for i in range(stack_layer):
        relFeature=highway(isshared)(relFeature)

    relFeature = Dropout(dropout)(relFeature)
    y = Dense(outputdim,activation=type)(relFeature)
    return y



def GCNLayer(multinetworks,contentFeature,inputdim,hiddendim,outputdim,_inception_depth,type='sigmoid',ishomo=False):
    if ishomo:
        multinetworks=[multinetworks[0]]

    relfeatures=[]
    for network in multinetworks:
        relfeature=contentFeature
        for i in range(_inception_depth-1):
            relfeature = tf.sparse_tensor_dense_matmul(network, relfeature)
            if i==0:
                W = tf.Variable(tf.random_uniform([inputdim, hiddendim], -1.0, 1.0))
            else:
                W = tf.Variable(tf.random_uniform([hiddendim, hiddendim], -1.0, 1.0))
            relfeature = Activation('relu')(K.dot(relfeature, W))
        relfeatures.append(relfeature)
    relfeature=merge(relfeatures, mode='concat', concat_axis=1) if not ishomo else relfeatures[0]

    y = Dense(outputdim, activation=type)(relfeature)
    return y



def HCC(multinetworks,contentFeature,labels,outputdim,type='sigmoid',ishomo=False):
    if ishomo:
        multinetworks=[multinetworks[0]]

    all_net_layer=[]
    for network in multinetworks:
        Net_feature = tf.sparse_tensor_dense_matmul(network, labels)
        all_net_layer.append(Net_feature)

    netlayer=merge(all_net_layer,mode='concat') if len(all_net_layer)>1 else all_net_layer[0]

    all_layer = merge([contentFeature, netlayer],mode='concat',concat_axis=1)
    y = Dense(outputdim, activation=type)(all_layer)
    return y