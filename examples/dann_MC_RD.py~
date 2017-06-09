'''
This is the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin

This allows domain adaptation (when you want to train on a dataset
with different statistics than a target dataset) in an unsupervised manner
by using the adversarial paradigm to punish features that help discriminate
between the datasets during backpropagation.

This is achieved by usage of the 'gradient reversal' layer to form
a domain invariant embedding for classification by an MLP.

The example here uses the 'MNIST-M' dataset as described in the paper.

Credits:
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.

Author: Vanush Vaswani (vanush@gmail.com)
'''

from __future__ import print_function
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.optimizers import SGD
from keras.models import Model
#from keras.utils.visualize_util import plot
from keras.utils import np_utils
import keras.backend as K

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#from sklearn.manifold import TSNE

from keras.layers import GradientReversal
from keras.engine.training import make_batches


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import time

import theano
import theano.tensor as T
import root_numpy

import keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Dropout
from keras.utils.np_utils import to_categorical
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score,accuracy_score
import pandas as pd
import pandas.core.common as com
from pandas.core.index import Index
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix


# Helper functions


def batch_gen(batches, id_array, data, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data[batch_ids], labels[batch_ids]
        else:
            yield data[batch_ids]
        np.random.shuffle(id_array)


def evaluate_dann(X_val, batch_size):
    """Predict batch by batch."""
    size = batch_size / 2
    num_batches = X_val.shape[0] / size
    acc = 0
    for i in range(0, num_batches):
        _, prob = dann_model.predict_on_batch(X_val[i * size:i * size + size])
        predictions = np.argmax(prob, axis=1)
        actual = np.argmax(y_val[i * size:i * size + size], axis=1)
        acc += float(np.sum((predictions == actual))) / size
    return acc / num_batches


# Model parameters

batch_size = 1000
nb_epoch = 5

_TRAIN = K.variable(1, dtype='uint8')


# Prep source data


events_pd = pd.DataFrame(root_numpy.root2array('/home/sebaleh/Documents/analysis/mc_tree/MC_PbPb_LHC16g1_MB_nanoAOD_compatible/pairtree_train_usnls.root','train_usnls',start=0,stop=100000,step=1))
events_pd = events_pd[events_pd['IsUS']>0.1]        #train only on US pairs

events_pd["IsSignal"]= ((events_pd["IsRP"]==1)& (events_pd["IsConv"]!=1)).values
events_pd["IsRPConv"]=((events_pd["IsRP"]==1)& (events_pd["IsConv"]==1)).values
events_pd["IsCombiwConv"]=((events_pd["IsRP"]==0)& (events_pd["IsConv"]==1)).values

y_combwconv = events_pd["IsCombiwConv"].values.astype(np.int32)
y_combwconv = y_combwconv-1
y_combwconv = y_combwconv/(-1)

X_pd = events_pd

X_calc = pd.DataFrame(X_pd["nITS1"])
X_calc["nITS2"] = X_pd["nITS2"]
X_calc["log DCAxy1"] = np.log(np.abs(X_pd["DCAxy1"]))
X_calc["log DCAxy2"] = np.log(np.abs(X_pd["DCAxy2"]))
X_calc["log DCAz1"] = np.log(np.abs(X_pd["DCAz1"]))
X_calc["log DCAz2"] = np.log(np.abs(X_pd["DCAz2"]))
X_calc["chi2GlobalpNDF1"] = X_pd["chi2GlobalpNDF1"]
X_calc["chi2GlobalpNDF2"] = X_pd["chi2GlobalpNDF2"]
X_calc["chi2ITS1"] = X_pd["chi2ITS1"]
X_calc["chi2ITS2"] = X_pd["chi2ITS2"] 
X_calc["HasSPDfirstHit_leg1"] = X_pd["HasSPDfirstHit_leg1"]
X_calc["HasSPDfirstHit_leg2"] = X_pd["HasSPDfirstHit_leg2"] 
X_calc["|opang|"] = np.abs(X_pd["opang"])
X_calc["mass"] = X_pd["mass"]
X_calc["phiv"] = X_pd["phiv"]
X_calc["diffz"] = X_pd["diffz"]
X_calc["pair_p"] = np.sqrt(   ((X_pd["px1"]+X_pd["px2"])*(X_pd["px1"]+X_pd["px2"])) + ((X_pd["py1"]+X_pd["py2"])*(X_pd["py1"]+X_pd["py2"])) + ((X_pd["pz1"]+X_pd["pz2"])*(X_pd["pz1"]+X_pd["pz2"]))   )
X_calc["delta_cot"] = np.abs( np.divide( X_pd["pz1"] , np.sqrt(X_pd["px1"]*X_pd["px1"] + X_pd["py1"]*X_pd["py1"])) - np.divide( X_pd["pz2"] , np.sqrt(X_pd["px2"]*X_pd["px2"] + X_pd["py2"]*X_pd["py2"])))
X_calc["eta1"] = X_pd["eta1"]
X_calc["eta2"] = X_pd["eta2"]
X_calc["phi1"] = X_pd["phi1"]
X_calc["phi2"] = X_pd["phi2"]
X_calc["pt1"] = X_pd["pt1"]
X_calc["pt2"] = X_pd["pt2"]
X = X_calc[X_calc.columns[:]].values.astype(np.float64)


X_train,X_val,y_train,y_val= train_test_split(X,y_combwconv,test_size=0.5,random_state=177)

y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

n_features = X_train.shape[1]


# Prep target data
events_pd = pd.DataFrame(root_numpy.root2array('/home/sebaleh/Documents/analysis/data_tree/LHC15o_nanoAOD_990/pairtree_usnls_small.root','pairtree_usnls',start=0,stop=100000,step=1))
events_pd = events_pd[events_pd['IsUS']>0.1]        #train only on US pairs

XT_pd = events_pd

XT_calc = pd.DataFrame(XT_pd["nITS1"])
XT_calc["nITS2"] = XT_pd["nITS2"]
XT_calc["log DCAxy1"] = np.log(np.abs(XT_pd["DCAxy1"]))
XT_calc["log DCAxy2"] = np.log(np.abs(XT_pd["DCAxy2"]))
XT_calc["log DCAz1"] = np.log(np.abs(XT_pd["DCAz1"]))
XT_calc["log DCAz2"] = np.log(np.abs(XT_pd["DCAz2"]))
XT_calc["chi2GlobalpNDF1"] = XT_pd["chi2GlobalpNDF1"]
XT_calc["chi2GlobalpNDF2"] = XT_pd["chi2GlobalpNDF2"]
XT_calc["chi2ITS1"] = XT_pd["chi2ITS1"]
XT_calc["chi2ITS2"] = XT_pd["chi2ITS2"] 
XT_calc["HasSPDfirstHit_leg1"] = XT_pd["HasSPDfirstHit_leg1"]
XT_calc["HasSPDfirstHit_leg2"] = XT_pd["HasSPDfirstHit_leg2"] 
XT_calc["|opang|"] = np.abs(XT_pd["opang"])
XT_calc["mass"] = XT_pd["mass"]
XT_calc["phiv"] = XT_pd["phiv"]
XT_calc["diffz"] = XT_pd["diffz"]
XT_calc["pair_p"] = np.sqrt(   ((XT_pd["px1"]+XT_pd["px2"])*(XT_pd["px1"]+XT_pd["px2"])) + ((XT_pd["py1"]+XT_pd["py2"])*(XT_pd["py1"]+XT_pd["py2"])) + ((XT_pd["pz1"]+XT_pd["pz2"])*(XT_pd["pz1"]+XT_pd["pz2"]))   )
XT_calc["delta_cot"] = np.abs( np.divide( XT_pd["pz1"] , np.sqrt(XT_pd["px1"]*XT_pd["px1"] + XT_pd["py1"]*XT_pd["py1"])) - np.divide( XT_pd["pz2"] , np.sqrt(XT_pd["px2"]*XT_pd["px2"] + XT_pd["py2"]*XT_pd["py2"])))
XT_calc["eta1"] = XT_pd["eta1"]
XT_calc["eta2"] = XT_pd["eta2"]
XT_calc["phi1"] = XT_pd["phi1"]
XT_calc["phi2"] = XT_pd["phi2"]
XT_calc["pt1"] = XT_pd["pt1"]
XT_calc["pt2"] = XT_pd["pt2"]
XT = XT_calc[XT_calc.columns[:]].values.astype(np.float64)


XT_train,XT_val= train_test_split(XT,test_size=0.5,random_state=177)

domain_labels = np.vstack([np.tile([0, 1], [batch_size / 2, 1]),
                           np.tile([1., 0.], [batch_size / 2, 1])])


class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Dense(50, activation='relu')(model_input)
        net = Dense(10, activation='relu')(net)
#        net = Flatten()(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu')(model_input)
        net = Dropout(0.5)(net)
        net = Dense(2, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, hp_lambda, plot_model=False):
        net = self._build_feature_extractor(main_input)
        self.grl = GradientReversal()
        branch = self.grl(net, hp_lambda)
        branch = Dense(50, activation='relu')(branch)
        branch = Dropout(0.1)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)

        # When building DANN model, route first half of batch (source examples)
        # to source classifier, and route full batch (half source, half target)
        # to the domain classifier.
        net = Lambda(lambda x: K.switch(K.learning_phase(), x[:int(batch_size / 2), :], x, lazy=True),         #if in training pass on first half (=MC) of batch else pass on both MC and RD
                     output_shape=lambda x: ((batch_size / 2,) + x[1:]))(net)                               #output shape??

        net = self._build_classifier(net)
        model = Model(input=main_input, output=[branch, net])
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

main_input = Input(shape=(X_train.shape[1],), name='main_input')

builder = DANNBuilder()
src_model = builder.build_source_model(main_input)
#src_vis = builder.build_tsne_model(main_input)

hp_lambda = K.variable(1)                                                     #adversiality trade off parameter
dann_model = builder.build_dann_model(main_input, hp_lambda)
#dann_vis = builder.build_tsne_model(main_input)
print('Training source only model')
src_model.fit(X_train, y_train, batch_size=1000, nb_epoch=10, verbose=1,
              validation_data=(X_val, y_val))
print('Evaluating target samples on source-only model')
print('Accuracy: ', src_model.evaluate(XT_val, y_val)[1])

# Broken out training loop for a DANN model.
src_index_arr = np.arange(X_train.shape[0])
target_index_arr = np.arange(XT_train.shape[0])

batches_per_epoch = len(X_train) / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0

print('Training DANN model')

for i in range(nb_epoch):

    batches = make_batches(X_train.shape[0], batch_size / 2)
    target_batches = make_batches(XT_train.shape[0], batch_size / 2)

    src_gen = batch_gen(batches, src_index_arr, X_train, y_train)
    target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)

    losses = list()
    acc = list()

    print('Epoch ', i)

    for (xb, yb) in src_gen:

        # Update learning rate and gradient multiplier as described in
        # the paper.
        p = float(j) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        hp_lambda = l
        builder.opt.lr = lr

        if xb.shape[0] != batch_size / 2:
            continue

        try:
            xt = target_gen.next()
        except:
            # Regeneration
            target_gen = target_gen(target_batches, target_index_arr, XT_train,
                                    None)

        # Concatenate source and target batch
        xb = np.vstack([xb, xt])

        metrics = dann_model.train_on_batch({'main_input': xb},
                                            {'classifier_output': yb,
                                            'domain_output': domain_labels},
                                            check_batch_dim=False)
        j += 1

print('Evaluating target samples on DANN model')
acc = evaluate_dann(XT_val, batch_size)
print('Accuracy:', acc)