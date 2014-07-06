import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from matplotlib import pyplot as pp

from models import LinearGaussian as LGmodel
from inference_engines import ImportanceSampler
from learning_algs import SGD_Momentum_Learner as SGDLearner

from collections import OrderedDict

statedims=4
datadims=20
nparticles=100

#trueM=np.sin(np.random.randn(statedims,datadims)*1000.0)*100.0
trueM=np.random.randn(statedims,datadims)
#trueM=np.arange(statedims*datadims).reshape((statedims,datadims))
true_model=LGmodel(statedims, datadims)
prop_model=LGmodel(datadims, statedims)

true_model.M.set_value(trueM.astype(np.float32))

true_posterior_cov_inv=np.dot(trueM,trueM.T)+np.eye(statedims)
true_posterior_cov=np.linalg.inv(true_posterior_cov_inv)
state=np.random.randn(statedims).astype(np.float32)
data=np.dot(state,trueM)+np.random.randn(datadims)
datashared=theano.shared(data)

print true_posterior_cov

prop_states=T.fmatrix()

prop_model.M.set_value(np.dot(true_posterior_cov,trueM).astype(np.float32).T)

truemean=np.dot(true_posterior_cov,np.dot(trueM,data.T))

def true_log_probs(prop_states):
	return true_model.rel_log_prob(prop_states,datashared) - 0.5*T.sum(prop_states**2,axis=1)

def sample_func():
	return prop_model.get_samples(T.extra_ops.repeat(datashared.dimshuffle('x',0),nparticles,axis=0))

sampler=ImportanceSampler(statedims,nparticles,true_log_probs,sample_func)
sampler.compile()

nsamps=100
samplestateshared=theano.shared(np.zeros((nsamps,statedims)).astype(np.float32))
sampledatashared=theano.shared(np.zeros((nsamps,datadims)).astype(np.float32))
sampleupdates=OrderedDict()
theano_rng=RandomStreams()
newstate=theano_rng.normal(size=(nsamps,statedims))
newdata=true_model.get_samples_noprobs(newstate)
sampleupdates[samplestateshared]=T.cast(newstate,'float32')
sampleupdates[sampledatashared]=T.cast(newdata,'float32')
sample_state_data=theano.function([],updates=sampleupdates)

prop_log_probs=prop_model.rel_log_prob(sampledatashared,samplestateshared,include_params_in_Z=True)
proploss=-T.mean(prop_log_probs)

proplearner=SGDLearner(prop_model.params,proploss,init_momentum_coeffs=[0.999], init_lrates=[1e-6], lrate_decay=0.9999)

esshist=[]
losshist=[]
for i in range(100):
	datashared.set_value((np.dot(state,trueM)+np.random.randn(datadims)).astype(np.float32))
	sampler.perform_sampling()
	esshist.append(sampler.get_ESS())
	for j in range(100):
		sample_state_data()
		proplearner.update_params()
		losshist.append(proplearner.compute_loss())

esshist=np.asarray(esshist)
losshist=np.asarray(losshist)
pp.plot(esshist)
pp.figure(2)
pp.plot(losshist)
pp.show()
exit()
print np.sum(sampler.weights.get_value())
#sampler.perform_resampling()
#print sampler.get_ESS()

particles=sampler.particles.get_value()
weights=sampler.weights.get_value()
sample_mean=np.dot(particles.T,weights)
pmm=particles-sample_mean
sample_cov=np.dot(pmm.T,pmm*weights.reshape((nparticles,1)))
print 'Sample mean / true posterior mean / true state'
print sample_mean
print truemean
print state
print 'Sample cov / true posterior cov'
print sample_cov
print true_posterior_cov
