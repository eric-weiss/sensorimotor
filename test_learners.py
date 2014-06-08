import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from matplotlib import pyplot as pp
import time

from learning_algs import SGD_Momentum_Learner


ndims=8000

init_x=np.random.randn(ndims)*2.0

c=np.random.randn(ndims)
M=np.random.randn(ndims, ndims)

shared_c=theano.shared(c.astype(np.float32))
shared_M=theano.shared(M.astype(np.float32))

shared_x=theano.shared(init_x.astype(np.float32))

Mx=T.dot(shared_M, shared_x)
loss=T.dot(shared_c.T, shared_x) + T.dot(Mx.T, Mx)

params=[shared_x]


nt=1000
lrate=[1e-6]
momentum_coeff=[0.95]

learner=SGD_Momentum_Learner(params, loss, momentum_coeff, lrate)
t0=time.time()
for i in range(nt):
	loss_now=learner.get_current_loss()
	learner.perform_learning_step()
	#print loss_now
print 'Time: ', time.time()-t0
print 'Final loss: ', loss_now

losshist=np.asarray(learner.loss_history)
pp.plot(losshist[:,1]); pp.show()
