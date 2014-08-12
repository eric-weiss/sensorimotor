import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SGD_Momentum_Learner():
	
	''' Implements stochastic gradient descent learning with basic 
	momentum.
	'''
	
	def __init__(self, params, loss, init_momentum_coeffs=[0.95], init_lrates=[1e-3], lrate_decay=0.9999999):
		''' 
		params: A list of parameters. They should be used to compute loss.
		
		loss: A symbolic function that computes the loss.
		
		init_momentum_coeffs: the initial momentum coefficients. It 
			should be a list. If the length is one, a single momentum
			coefficient is used for every parameter. It could also have
			length equal to the number of parameters in model, in which 
			case each element will correspond to a parameter in model.
			
		init_lrates: initial learning rates. Works the same as 
			init_momentum_coeffs, see above.
		
		lrate_decay: Every time learning is performed, lrates are multiplied
			by this number.
		'''
		
		self.lrate_decay=np.float32(lrate_decay)
		self.n_params=len(params)
		self.params=params
		self.loss=loss
		
		if len(init_momentum_coeffs)==1:
			init_momentum_coeffs=init_momentum_coeffs[0]*np.ones(self.n_params)
		
		if len(init_lrates)==1:
			init_lrates=init_lrates[0]*np.ones(self.n_params)
		
		self.momentum_coeffs=init_momentum_coeffs.astype(np.float32)
		self.lrates=init_lrates.astype(np.float32)
		
		self.global_lrate=theano.shared(np.float32(1.0))
		
		#the momentums for each parameter in self.params are stored as a list
		self.momentums=[]
		for param in params:
			init_momentum=np.zeros_like(param.get_value()).astype(np.float32)
			self.momentums.append(theano.shared(init_momentum))
		
		#every time get_current_loss is called, the loss, along with the 
		#current iteration number, is appended to this list as an array:
		#[iteration number, loss]
		self.loss_history=[]
		
		self.n_learning_iterations=0
		
		#this function computes the current loss
		self.compute_loss=theano.function([],self.loss)
		
		self.loss_history.append([0, self.compute_loss()])
		
		#calling this function executes one step of learning
		learn_updates=self.learn_step()
		self.update_params=theano.function([],updates=learn_updates)
	
	
	def learn_step(self):
		
		#this is a list of gradients w.r.t. every parameter in self.params
		gparams=T.grad(self.loss, self.params)
		
		updates=OrderedDict()
		#updates the momentums and parameter values
		i=0
		for param, gparam, momentum, lrate, momentum_coeff in zip(self.params, gparams, self.momentums, self.lrates, self.momentum_coeffs):
			
			#if param.ndim==2:
			#	gparam=T.dot(T.dot(param,param.T),gparam)
			
			if param.name=='log_stddev':
				gparam=gparam*2.0*T.exp(4.0*param)
			
			if param.name=='M':
				gparam=gparam*T.exp(2.0*self.params[i+2]).dimshuffle('x',0)
			
			if param.name=='b':
				gparam=gparam*T.exp(2.0*self.params[i+1])
			
			new_momentum=momentum_coeff*momentum - lrate*gparam*self.global_lrate
			new_param=param + new_momentum
			
			updates[param]=new_param
			updates[momentum]=new_momentum
			i+=1
		
		updates[self.global_lrate]=self.global_lrate*self.lrate_decay
		
		return updates
	
	
	def get_current_loss(self):
		
		'''This computes and returns the current loss.
		'''
		current_loss=self.compute_loss()
		self.loss_history.append([self.n_learning_iterations, current_loss])
		return current_loss
	
	
	def perform_learning_step(self):
		
		self.update_params()
		self.n_learning_iterations+=1
		return

