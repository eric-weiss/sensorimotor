import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class ParticleFilter():
	''' Implements particle filtering and smoothing for arbitrary proposal/true distributions '''
	
	def __init__(self, data_dims, state_dims, n_particles, n_history=1, resample_thresh=0.5):
		
		
		
		self.data_dims=data_dims
		self.state_dims=state_dims
		self.n_particles=n_particles
		self.resample_thresh=resample_thresh
		self.n_history=n_history
		
		#this is used to keep track of what set of particles corresponds
		#to the previous point in time
		self.time_counter=theano.shared(0)
		
		#self.particle_history=[]
		#self.weight_history=[]
		
		#for i in range(n_history+1):
			#init_particles=np.zeros((n_particles, state_dims)).astype(np.float32)
			#init_weights=(np.ones(n_particles)/float(n_particles)).astype(np.float32)
			#particle_history.append(theano.shared(init_particles))
			#weight_history.append(theano.shared(init_weights))
		
		init_particles=np.zeros((n_history+1, n_particles, state_dims)).astype(np.float32)
		init_weights=(np.ones((n_history+1, n_particles))/float(n_particles)).astype(np.float32)
		
		self.particles=theano.shared(init_particles)
		self.weights=theano.shared(init_weights)
		
		self.current_state=self.particles[self.time_counter%self.n_history]
		self.previous_state=self.particles[(self.time_counter-1)%self.n_history]
		
		self.proposal_samples=None
		self.log_proposal_probs=None
		
		self.true_log_transition_probs=None
		self.true_log_observation_probs=None
	
	
	def set_proposal(self, proposal_samples, log_probs):
		
		self.proposal_samples=proposal_samples
		self.log_probs=log_probs
		return
	
	
	def set_true_log_probs(self, true_log_probs):
		
		self.true_log_probs=true_log_probs
		return
	
	
	
		
	
	
	
