#encoding=utf-8
from keras.layers import ActivityRegularization
from metric.metrics import *
from models.Models import *
from models.inception import *


result=open('results/test.txt','a')
dataname='ACMdataset'



para={'algorithm':'ourlayer', 'output_type':'softmax','linktype':'meanlinks',   ##describe the experiment
       'stack_layer':5, 'ishighway':True,'isshared':True,                      ##stack
       '_kernel_size':2,'_inception_depth':4, 'hiddennum':16,                   ##convolution paramater
        'dropout': 0.5, 'ispart':False                           ##basic paramater
     }



if 'DBLP' in dataname:
    para['output_type'] = 'softmax'
    para['ispart'] = True
elif 'IMDB' in dataname:
    para['output_type'] = 'sigmoid'
elif 'ACM' in dataname:
    para['output_type'] = 'softmax'
elif 'SLAP' in dataname:
    para['output_type'] = 'softmax'
    para['ispart'] = True


truelabels,truefeatures,rownetworks,knownindex=getData(dataname,para['linktype'],para['ispart'])
samples,labelnums,featurenums=truelabels.shape[0],truelabels.shape[1],truefeatures.shape[1]
iternum,epochs,perpass,iterica,Kholdoutvalidation=10.0,60,25,10,5
###########################################################################################

para['_inception_depth'] = 4
para['_kernel_size'] = 2
para['hiddennum'] = labelnums *4
para['algorithm'] = 'StackedLearning'
if 'GCN' in para['algorithm']:
    para['linktype']='gcnlinks'
else:
    para['linktype'] = 'meanlinks'


###########################################################################################
allnetworks=[]
for networks in rownetworks:
    tmp = scipy.sparse.csr_matrix(networks)
    coords, values, shape = sparse_to_tuple(tmp)
    allnetworks.append(tf.SparseTensorValue(coords, values, shape))
for numi in range(int(iternum)):
    ######################################################################################
    # feed dicts and initalize session
    index = np.random.randint(0, Kholdoutvalidation, (samples, 1)) > 0
    trainindex, testindex = np.where(index == True)[0], np.where(index == False)[0]
    if para['ispart']:
        trainindex=list(set(knownindex).intersection(trainindex))
        testindex=list(set(knownindex).intersection(testindex))

    testlabels = truelabels.copy()
    testlabels[testindex] = 0
    #####################################################################################
    #input layer
    labels=tf.placeholder('float',[None,labelnums])
    features=tf.placeholder('float',[None,None])
    Net=[tf.sparse_placeholder('float', [None,None]) for _ in range(len(allnetworks))]
    isstop=tf.placeholder('bool')
    static_feature = tf.reshape(features, [-1, featurenums])
    select_index = tf.placeholder('int32', [None])
    #####################################################################################
    #compute layer
    #################################################################
    if para['algorithm'] is 'LR':
        y=Dense(labelnums, activation=para['output_type'])(static_feature)
    #################################################################
    elif para['algorithm'] is 'HighwayNet':
        y=HighwayNet(static_feature,para['hiddennum'],labelnums,para['stack_layer'],
                                  type=para['output_type'],isshared=para['isshared'],dropout=0.5)
    #################################################################
    elif para['algorithm'] is 'HCC':
        y=HCC(Net,static_feature,labels,labelnums,type=para['output_type'],ishomo=False)
    #################################################################
    elif para['algorithm'] is 'ICA':
        y = HCC(Net, static_feature, labels, labelnums, type=para['output_type'], ishomo=True)
    #################################################################
    elif para['algorithm'] is 'GraphInception_ICA':
        y = GraphInception(Net, static_feature, labels, labelnums, labelnums,para['_inception_depth'],para['_kernel_size'],
                           para['hiddennum'],outputtype=para['output_type'])
    #################################################################
    elif para['algorithm'] is 'GraphInception_Stack':
        y=StackInception(Net, static_feature, labelnums,para['_inception_depth'],para['_kernel_size'], para['hiddennum'],outputtype=para['output_type'])
    #################################################################
    elif para['algorithm'] is 'CLN':
        y=CLN(Net,static_feature,para['hiddennum'],labelnums,para['stack_layer'],dropout=0.5,
                           type=para['output_type'],ishighway=para['ishighway'],isshared=para['isshared'])
    #################################################################
    elif para['algorithm'] is 'StackedLearning':
        y=StackLearning(Net,static_feature,labelnums,para['_inception_depth'],type=para['output_type'])
    #################################################################
    elif para['algorithm'] is 'GCN':
        y = GCNLayer(Net,static_feature,featurenums,para['hiddennum'],labelnums,
                                  para['_inception_depth'],type=para['output_type'],ishomo=True)
    #################################################################
    elif para['algorithm'] is 'GCN_metapath':
        y = GCNLayer(Net, static_feature, featurenums, para['hiddennum'], labelnums,
                                   para['_inception_depth'], type=para['output_type'], ishomo=False)
    #################################################################
    else:
        print 'There doesn\'t exists the algorithm.'

    y=ActivityRegularization(l1=0.01,l2=0.01)(y)
    #####################################################################################
    if 'GCN' in para['algorithm']:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.gather(y,select_index),tf.gather(labels,select_index)))
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,labels))

    train = tf.train.RMSPropOptimizer(0.01).minimize(loss)

    if 'GCN' in para['algorithm']:
        traindicts = {labels: testlabels, features: truefeatures, select_index: trainindex,K.learning_phase():1}
        traindicts=dict(traindicts.items()+{Net[i]: allnetworks[i] for i in range(len(allnetworks))}.items())
    else:
        tmpnetworks = []
        for rownetwork in rownetworks:
            coords, values, shape = sparse_to_tuple(rownetwork[trainindex, :][:, trainindex])
            tmpnetworks.append(tf.SparseTensorValue(coords, values, shape))
        traindicts = {labels: truelabels[trainindex], features: truefeatures[trainindex], select_index: trainindex,K.learning_phase(): 1}
        traindicts = dict(traindicts.items() + {Net[i]: tmpnetworks[i] for i in range(len(allnetworks))}.items())

    testdicts = {labels: testlabels, features: truefeatures, select_index: testindex,K.learning_phase():0}
    testdicts = dict(testdicts.items() + {Net[i]: allnetworks[i] for i in range(len(allnetworks))}.items())
    ################################################################################################################
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #########################################################################################################
    print para
    for step in range(epochs):
        [sess.run(train, feed_dict=traindicts) for iter in range(perpass)] ##train 40 epochs each step
        testdicts[labels][testindex] = 0
        if 'ICA' in para['algorithm'] or 'HCC' in para['algorithm']:
            for i in range(iterica):
                testdicts[labels][trainindex]=truelabels[trainindex]
                testlabels=sess.run(y, feed_dict=testdicts)
        else:
            testlabels = sess.run(y, feed_dict=testdicts)

    ############################evaluate results################################################
        fscore_macro = fscore(truelabels[testindex], testlabels[testindex], type='macro')
        hamming_loss = hamming_distance(truelabels[testindex], testlabels[testindex])
        accuracy_s = accuracy_subset(truelabels[testindex], testlabels[testindex])
        accuracy_class=accuracy_multiclass(truelabels[testindex], testlabels[testindex])
        fscore_sa = fscore_class(truelabels[testindex], testlabels[testindex], type='macro')
        print step,'train',fscore(truelabels[trainindex],testlabels[trainindex],type='macro'),\
            hamming_distance(truelabels[trainindex], testlabels[trainindex]),\
            accuracy_subset(truelabels[trainindex], testlabels[trainindex]),\
            accuracy_multiclass(truelabels[trainindex], testlabels[trainindex])
        print step,'test',fscore_macro,hamming_loss,accuracy_s,accuracy_class,fscore_sa
    ################################################################################################################
        result.write(str(step*perpass)+':'+str([para, fscore_macro, hamming_loss, accuracy_s,accuracy_class,fscore_sa]) + '\n')

result.close()
