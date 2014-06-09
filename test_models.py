import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
import time

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from models import LinearGaussian as Lmodel

from matplotlib import pyplot as pp

inputdims=2
outputdims=4
log_stddev=np.random.randn(outputdims)

model=Lmodel(inputdims,outputdims)

model.M.set_value(np.zeros((inputdims, outputdims)).astype(np.float32))
model.log_stddev.set_value(log_stddev.astype(np.float32))

inputsamps=T.fmatrix()
samps, probs = model.get_samples(inputsamps)
sample=theano.function([inputsamps],[samps, probs],allow_input_downcast=True)

s=np.zeros((10000,2))
samples, log_probs = sample(s)

var=np.mean(samples**2,axis=0)
print var
print np.exp(2.0*log_stddev)
