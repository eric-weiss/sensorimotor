import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from matplotlib import pyplot as pp

from models import LinearGaussian as LGmodel
from inference_engines import ImportanceSampler

statedims=1
datadims=10
nparticles=200

trueM=np.ones((statedims,datadims))*10.0

true_model=LGmodel(statedims, datadims)
prop_model=LGmodel(datadims, statedims)

true_model.M.set_value(trueM.astype(np.float32))

true_posterior_cov_inv=np.dot(trueM,trueM.T)+np.eye(statedims)
true_posterior_cov=np.linalg.inv(true_posterior_cov_inv)
state=np.random.randn(statedims).astype(np.float32)
data=np.dot(state,trueM)+np.random.randn(datadims)
datashared=theano.shared(data)
prop_states=T.fmatrix()

prop_model.M.set_value(np.dot(true_posterior_cov,trueM).astype(np.float32).T)

truemean=np.dot(true_posterior_cov,np.dot(trueM,data.T))

def true_log_probs(prop_states):
	return true_model.rel_log_prob(prop_states,datashared) - 0.5*T.sum(prop_states**2,axis=1)

def sample_func():
	return prop_model.get_samples(T.extra_ops.repeat(datashared.dimshuffle('x',0),nparticles,axis=0))

sampler=ImportanceSampler(statedims,nparticles,true_log_probs,sample_func)
sampler.compile()

sampler.perform_sampling()
print sampler.get_ESS()
print np.sum(sampler.weights.get_value())
#sampler.perform_resampling()
#print sampler.get_ESS()

print np.sum(sampler.particles.get_value()*sampler.weights.get_value().reshape((nparticles,statedims)),axis=0)
print truemean
print state

