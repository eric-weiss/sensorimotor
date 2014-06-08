import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
import time

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from models import LinearGaussian as Lmodel

from matplotlib import pyplot as pp

from inference_engines import ParticleFilter

statedims=2
datadims=10
nparticles=1000

PF=ParticleFilter(datadims, statedims, nparticles, n_history=100)

genproc=Lmodel(statedims, datadims)
tranproc=Lmodel(statedims, statedims)

nt=1000

theta=0.1
trueM=np.asarray([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]],dtype='float32')
trueG=(np.random.randn(statedims,datadims)*0.5).astype(np.float32)
tranproc.M.set_value(trueM)
genproc.M.set_value(trueG)
tranproc.log_stddev.set_value((np.ones(statedims)*-4.0).astype(np.float32))

s=[np.asarray([0,1])]
for i in range(nt):
	s.append(np.dot(s[i],trueM))
s=np.asarray(s)
xsamps=np.dot(s,trueG)+np.random.randn(s.shape[0],datadims)

#M=tranproc.Ms.get_value()
#sp=tranproc.spreads.get_value()
#c=tranproc.centers.get_value()
#b=tranproc.biases.get_value()
#M[0]=-2.0*np.eye(statedims)
#sp[0]=sp[0]*0.1
#c[0]=c[0]*0.0
#b[0]=-16.0
#tranproc.Ms.set_value(M)
#tranproc.spreads.set_value(sp)
#tranproc.centers.set_value(c)
#tranproc.biases.set_value(b)

#s=T.fmatrix()
#snext=theano.function([s],tranproc.compute_conditional_means(s)*0.01+s,allow_input_downcast=True)


#st=np.random.randn(1,statedims)
#sh=[st[0]]
#for i in range(nt):
	#st=snext(sh[i].reshape((1,statedims)))
	#sh.append(st[0])
#sh=np.asarray(sh)
#pp.plot(sh)
#pp.show()
#exit()
	

prop_samps, prop_probs = tranproc.get_samples(PF.current_state)

PF.set_proposal(prop_samps, prop_probs)

PF.set_true_log_observation_probs(genproc.rel_log_prob)
PF.set_true_log_transition_probs(tranproc.rel_log_prob)

PF.recompile()

#pfupdates = PF.sample_update(x)
#samplestep=theano.function([x],[],updates=pfupdates,allow_input_downcast=True)

#resupdates=PF.resample()
#resample=theano.function([],[],updates=resupdates)

#getESS=theano.function([],PF.get_ESS())

getstates=theano.function([],PF.current_state)
getweights=theano.function([],PF.current_weights)


esshist=[]
statehist=[]
weighthist=[]
t0=time.time()
for i in range(nt):
	PF.perform_inference(xsamps[i])
	ess=PF.get_ESS()
	esshist.append(ess)
	if ess<nparticles/4:
		PF.resample()
	statehist.append(getstates())
	weighthist.append(getweights())

print time.time()-t0
esshist=np.asarray(esshist)
statehist=np.asarray(statehist)
weighthist=np.asarray(weighthist)
jointsamples=np.asarray(PF.sample_joint(100))
meanjoint=np.mean(jointsamples,axis=1)

meanstates=np.sum(statehist*np.reshape(weighthist, (statehist.shape[0], nparticles, 1)), axis=1)

pp.plot(esshist)
pp.figure(2)
pp.plot(meanstates)
pp.plot(s)
pp.figure(3)
pp.plot(meanjoint)
pp.plot(s[-100:])
pp.show()





