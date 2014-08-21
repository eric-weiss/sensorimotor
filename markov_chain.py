import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
from matplotlib import pyplot as pp

from models import LinearGaussian as Lmodel
from models import MixedLinearGaussian as MLmodel
from inference_engines import ParticleFilter
from learning_algs import SGD_Momentum_Learner as SGDLearner

statedims=2
datadims=8
nparticles=200

n_joint_samples=128
n_history=100

nt=4000

#======Making data=======================

#Linear/Gaussian
antisym=np.tril(np.ones((statedims, statedims)),k=-1); antisym=antisym-antisym.T
trueM=np.eye(statedims,k=1)*32e-2
trueM=trueM+trueM.T; trueM=trueM*antisym+np.eye(statedims)

#trueG=np.sin(np.reshape(np.arange(statedims*datadims),(statedims,datadims))*10.0)*2.0
#trueG[0,:datadims/2]=0.0
#trueG[1,datadims/2:]=0.0

trueG=MLmodel(statedims,datadims,4)
trueG.log_stddev.set_value((np.ones(datadims)*-4.0).astype(np.float32))
inputT=T.fmatrix()
outputT=trueG.get_samples_noprobs(inputT)
sampleG=theano.function([inputT],outputT,allow_input_downcast=True)

true_log_stddev=np.zeros(statedims)-10.0

s0=np.zeros(statedims); s0[0]=4.0; s0=s0.astype(np.float32)
true_s=[s0]
for i in range(nt):
	next_s=np.dot(true_s[i],trueM)
	next_s=4.0*next_s/np.sqrt(np.sum(next_s**2))
	true_s.append(next_s+np.random.randn(statedims)*np.exp(true_log_stddev))

#Nonlinear/Gaussian
#true_model=MLmodel(statedims,datadims,3)
#s0=np.zeros(statedims)
#s0[0]=4.0
#s0=s0.astype(np.float32)
#true_s=[s0]
#inputT=T.fmatrix()

#for i in range(nt):
	

true_s=np.asarray(true_s,dtype='float32')
#observations=np.dot(true_s,trueG)+np.random.randn(true_s.shape[0],datadims)*np.exp(-4.0)
observations=sampleG(true_s)

#pp.plot(true_s)
#pp.figure(2)
pp.plot(observations)
pp.show()

shared_obs=theano.shared(observations.astype(np.float32))
shared_t=theano.shared(n_history+2)
current_observation=shared_obs[shared_t]
learning_observations=shared_obs[shared_t-n_history:shared_t+1]
increment_t=theano.function([],updates={shared_t: shared_t+1})
#========================================

genproc=MLmodel(statedims, datadims,8)
tranproc=Lmodel(statedims, statedims)

#genproc.M.set_value((genproc.M.get_value()*1e0).astype(np.float32))
#genproc.M.set_value(trueG.astype(np.float32))
#genproc.log_stddev.set_value((np.ones(datadims)*-1).astype(np.float32))

proposal_model=Lmodel(statedims+datadims,statedims)

#proposal_model.M.set_value((np.concatenate([np.eye(statedims)*1e-2,genproc.M.get_value().T],axis=0)).astype(np.float32))
proposal_model.log_stddev.set_value((np.ones(statedims)*1.0).astype(np.float32))

PF=ParticleFilter(tranproc, genproc, nparticles, n_history=n_history, observation_input=current_observation)


tranproc.M.set_value(np.eye(statedims).astype(np.float32))
#tranproc.M.set_value(trueM.astype(np.float32))
tranproc.log_stddev.set_value((np.ones(statedims)*1.0).astype(np.float32))


prop_distrib = proposal_model.get_samples(T.concatenate(
		[PF.current_state, T.extra_ops.repeat(current_observation.dimshuffle('x',0),nparticles,axis=0)],axis=1))

PF.set_proposal(prop_distrib)
PF.recompile()

#future_model_data, future_model_states, future_init_states, future_updates=PF.sample_future(69,1)

#proposal_loss=-T.mean(proposal_model.rel_log_prob(T.concatenate([future_init_states,future_model_data[0]],axis=1),
			#future_model_states[0],include_params_in_Z=True))

n_prop_samps=64
n_prop_T=10


#W=genproc.M.get_value()
#sigx=np.exp(-2.0*genproc.log_stddev.get_value()).reshape((datadims,1))
#M=tranproc.M.get_value()
#sigs=np.exp(-2.0*tranproc.log_stddev.get_value()).reshape((statedims,1))
#init_prop_log_stddev=-0.5*np.log(np.diag(np.dot(W,sigx*W.T)+sigs*np.eye(statedims)))


total_params=tranproc.params + genproc.params
#total_params=[tranproc.M, genproc.M]#, tranproc.log_stddev]

obs=T.fvector()
shared_joint_samples=theano.shared(np.zeros(((n_history+1)*n_joint_samples, statedims)).astype(np.float32))

joint_samples, joint_sample_updates=PF.sample_from_joint(n_joint_samples,output_2D=True)
joint_sample_updates[shared_joint_samples]=joint_samples
perform_joint_sampling=theano.function([],updates=joint_sample_updates)

tranloss=T.mean(tranproc.rel_log_prob(shared_joint_samples[:-n_joint_samples],shared_joint_samples[n_joint_samples:],include_params_in_Z=True))
genloss=T.mean(genproc.rel_log_prob(shared_joint_samples,T.extra_ops.repeat(learning_observations,n_joint_samples,axis=0),include_params_in_Z=True))
total_loss=-(tranloss+genloss)


init_lrates=np.asarray([1e-2, 1e-3, 1e-3, 1e-2, 0e-2, 0e-2, 1e-2, 1e-3, 1e-3])*2e0

learner=SGDLearner(total_params, total_loss, init_lrates=init_lrates,init_momentum_coeffs=[0.1])


proplogprobs=proposal_model.rel_log_prob(T.concatenate([PF.previous_state,
						T.extra_ops.repeat(current_observation.dimshuffle('x',0),nparticles,axis=0)],axis=1),
							PF.current_state,include_params_in_Z=True,all_pairs=True)

proposal_loss=-T.dot(PF.previous_weights,T.dot(proplogprobs,PF.current_weights))
#proposal_loss=-T.mean(proplogprobs)

proposal_learner=SGDLearner(proposal_model.params,proposal_loss,init_lrates=np.asarray([1e-3, 1e-4, 1e-4])*4e0,init_momentum_coeffs=[0.2])



print 'Done compiling, beginning training'
esshist=[]
t0=time.time()
statehist=[]
weighthist=[]
paramhist=[]
proplosshist=[]
min_learn_delay=80
learn_counter=0
nan_occurred=False
preddatahist=[]
#proposal_learner.global_lrate.set_value(np.float32(200.0))
for i in range(nt-1000):
	PF.perform_inference()
	ess=PF.get_ESS()
	esshist.append(ess)
	#print ess
	#pp.plot(PF.get_current_weights())
	#pp.show()
	if ess<nparticles/4 or True:
		ess,stddevhist,esshs=PF.perform_sequential_resampling()
		if np.isnan(ess):
			pp.plot(stddevhist)
			pp.figure(2)
			pp.plot(esshs)
			pp.show()
	#print ess
	statehist.append(PF.get_current_particles())
	weighthist.append(PF.get_current_weights())
	paramhist.append(genproc.log_stddev.get_value())
	ess=PF.get_ESS()
	
	for j in range(10):
		proposal_learner.perform_learning_step()
	#print '====================='
	#print PF.get_current_particles()
	#print proposal_model.M.get_value()
	#print proposal_model.b.get_value()
	#print proposal_model.log_stddev.get_value()
	#print '====================='
	#print ess
	
	#pp.figure(2)
	#pp.plot(esshs)
	#pp.show()
	#esshist.append(newess)
	if learn_counter>min_learn_delay:# and ess>nparticles/32:
		perform_joint_sampling()
		#pp.plot(stddevhist)
		#pp.show()
		#js=shared_joint_samples.get_value()
		#pp.plot(np.mean(js.reshape((n_history+1,n_joint_samples,statedims)),axis=1))
		#pp.figure(2)
		#print np.asarray(statehist[-(n_history+1):]).shape
		#pp.plot(np.asarray(statehist[-(n_history+1):]).reshape(((n_history+1),statedims*nparticles)))
		#pp.show()
		loss0=learner.get_current_loss()
		if np.isnan(loss0):
			nan_occurred=True
			break
		for j in range(200):
			learner.perform_learning_step()
		#loss1=learner.get_current_loss()
		learn_counter=0
		#for j in range(10):
			##update_model_samples()
			#proposal_learner.perform_learning_step()
			#proplosshist.append(proposal_learner.get_current_loss())
		
		print 'Iteration ', i
		print 'Loss: ', loss0, '  Proploss: ', proposal_learner.get_current_loss()
		print tranproc.M.get_value()
		print trueM
		print tranproc.log_stddev.get_value()
	
	statesamps=PF.sample_current_state(100)
	#preddatahist.append(np.mean(np.dot(statesamps,genproc.M.get_value()),axis=0))
	preddatahist.append(np.mean(genproc.sample_output(statesamps),axis=0))
	
		
	if nan_occurred:
		break
	if ess<nparticles*0.5:
		PF.resample()
	
	increment_t()
	
	learn_counter+=1

print tranproc.M.get_value()
print trueM
print tranproc.log_stddev.get_value()
print true_log_stddev
print genproc.log_stddev.get_value()

futuresamps,futurestates,futureinit=PF.sample_from_future(10,1000)
futuremeans=np.mean(futuresamps,axis=1)
	
statehist=np.asarray(statehist,dtype='float32')
weighthist=np.asarray(weighthist,dtype='float32')
esshist=np.asarray(esshist,dtype='float32')
proplosshist=np.asarray(proplosshist,dtype='float32')

meanstate=np.sum(statehist*np.reshape(weighthist,(statehist.shape[0],statehist.shape[1],1)),axis=1)

losshist=np.asarray(learner.loss_history)
pp.plot(losshist[:,1])
pp.figure(2)
pp.plot(futuremeans,'r')
#pp.plot(futuresamps.reshape((1000,10*datadims)),'r')
pp.plot(observations[shared_t.get_value():],'b')
pp.figure(3)
pp.plot(meanstate,'r')
pp.plot(true_s[n_history+2:shared_t.get_value()],'b')
pp.figure(4)
pp.plot(esshist)
pp.figure(5)
print shared_t.get_value()-(n_history+2)
pp.plot(np.concatenate([preddatahist,futuremeans]),'r')
pp.plot(observations[n_history+2:],'b')
pp.figure(6)
pp.plot(paramhist)
pp.show()
