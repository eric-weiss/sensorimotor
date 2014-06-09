import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
from matplotlib import pyplot as pp

from models import LinearGaussian as Lmodel
from inference_engines import ParticleFilter
from learning_algs import SGD_Momentum_Learner as SGDLearner

statedims=5
datadims=10
nparticles=1000

n_joint_samples=10

nt=10000

#======Making data=======================
antisym=np.tril(np.ones((statedims, statedims)),k=-1); antisym=antisym-antisym.T
trueM=np.random.randn(statedims,statedims)*4e-2
trueM=np.dot(trueM, trueM.T); trueM=trueM*antisym+np.eye(statedims)
trueG=np.random.randn(statedims,datadims)
true_log_stddev=np.random.randn(statedims)-10.0

s0=np.zeros(statedims); s0[0]=1.0; s0=s0.astype(np.float32)
true_s=[s0]
for i in range(nt):
	next_s=np.dot(true_s[i],trueM)
	next_s=next_s/np.sqrt(np.sum(next_s**2))
	true_s.append(next_s+np.random.randn(statedims)*np.exp(true_log_stddev))
true_s=np.asarray(true_s,dtype='float32')
observations=np.dot(true_s,trueG)+np.random.randn(true_s.shape[0],datadims)

pp.plot(true_s)
pp.show()

shared_obs=theano.shared(observations.astype(np.float32))
shared_t=theano.shared(0)
current_observation=shared_obs[shared_t]
increment_t=theano.function([],updates={shared_t: shared_t+1})
#========================================

PF=ParticleFilter(datadims, statedims, nparticles, n_history=1, observation_input=current_observation)

genproc=Lmodel(statedims, datadims)
tranproc=Lmodel(statedims, statedims)

tranproc.M.set_value(np.eye(statedims).astype(np.float32))
tranproc.log_stddev.set_value((np.ones(statedims)*-2.0).astype(np.float32))

prop_samps, prop_probs = tranproc.get_samples(PF.current_state)

PF.set_proposal(prop_samps, prop_probs)

PF.set_true_log_observation_probs(genproc.rel_log_prob)
PF.set_true_log_transition_probs(tranproc.rel_log_prob)

PF.recompile()

#total_params=tranproc.params + genproc.params
total_params=[tranproc.M, genproc.M]

obs=T.fvector()
joint_samples, joint_sample_updates=PF.sample_from_joint(n_joint_samples)
tranloss=T.mean(tranproc.rel_log_prob(joint_samples[0],joint_samples[1],include_params_in_Z=True))
genloss=T.mean(genproc.rel_log_prob(joint_samples[1],current_observation.dimshuffle('x',0),include_params_in_Z=True))
total_loss=-(tranloss+genloss)
#lrates=np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])*2e-4
lrates=np.asarray([1.0, 1.0])*1e-0

#learner=SGDLearner(total_params, total_loss, init_lrates=lrates)
learner=SGDLearner(total_params, total_loss, init_lrates=[1e-3])




print 'Done compiling, beginning training'
esshist=[]
t0=time.time()
learn_every=10
for i in range(nt):
	PF.perform_inference()
	ess=PF.get_ESS()
	esshist.append(ess)
	
	if (i+1)%learn_every==0:
		learner.perform_learning_step()
		loss=learner.get_current_loss()
		print loss
	
	if ess<nparticles/8:
		PF.resample()
	
	increment_t()

losshist=np.asarray(learner.loss_history)
pp.plot(losshist[:,1])
pp.figure(2)
pp.plot(np.asarray(esshist))
pp.show()
