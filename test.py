import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from models import LinearGaussian

from matplotlib import pyplot as pp

from inference_engines import ParticleFilter

statedims=2
datadims=5
nparticles=1000

PF=ParticleFilter(datadims, statedims, nparticles)

genproc=LinearGaussian(statedims, datadims)
tranproc=LinearGaussian(statedims, statedims)

theta=0.1
trueM=np.asarray([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]],dtype='float32')
trueG=(np.random.randn(statedims,datadims)*0.5).astype(np.float32)
tranproc.M.set_value(trueM)
genproc.M.set_value(trueG)
tranproc.log_stddev.set_value((np.ones(statedims)*-5.0).astype(np.float32))

nt=10000
s=[np.asarray([0,1])]
for i in range(nt):
	s.append(np.dot(s[i],trueM))
s=np.asarray(s)
xsamps=np.dot(s,trueG)+np.random.randn(s.shape[0],datadims)

prop_samps, prop_probs = tranproc.get_samples(PF.current_state)

PF.set_proposal(prop_samps, prop_probs)

PF.set_true_log_observation_probs(genproc.rel_log_prob)
PF.set_true_log_transition_probs(tranproc.rel_log_prob)

x=T.fvector()

pr=PF.true_log_observation_probs(PF.proposal_samples, x.dimshuffle('x',0))


pfupdates = PF.sample_update(x)
samplestep=theano.function([x],[],updates=pfupdates,allow_input_downcast=True)

resupdates=PF.resample()
resample=theano.function([],[],updates=resupdates)

getESS=theano.function([],PF.get_ESS())

getstates=theano.function([],PF.current_state)
getweights=theano.function([],PF.current_weights)


esshist=[]
statehist=[]
weighthist=[]
for i in range(nt):
	samplestep(xsamps[i])
	ess=getESS()
	esshist.append(ess)
	if ess<nparticles/2:
		resample()
	statehist.append(getstates())
	weighthist.append(getweights())

esshist=np.asarray(esshist)
statehist=np.asarray(statehist)
weighthist=np.asarray(weighthist)

meanstates=np.sum(statehist*np.reshape(weighthist, (statehist.shape[0], nparticles, 1)), axis=1)

pp.plot(esshist)
pp.figure(2)
pp.plot(meanstates)
pp.show()





