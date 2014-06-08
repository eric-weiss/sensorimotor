import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class ParticleFilter():
	''' Implements particle filtering and smoothing for Markov Chains
	 with arbitrary proposal/true distributions '''
	
	def __init__(self, data_dims, state_dims, n_particles, n_history=1):
		
		self.data_dims=data_dims
		self.state_dims=state_dims
		self.n_particles=n_particles
		self.n_history=n_history
		
		#this is used to keep track of what set of particles corresponds
		#to the previous point in time
		self.time_counter=theano.shared(0)
		
		self.theano_rng=RandomStreams()
		
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
		
		self.next_state=self.particles[(self.time_counter+1)%(self.n_history+1)]
		self.current_state=self.particles[self.time_counter%(self.n_history+1)]
		self.previous_state=self.particles[(self.time_counter-1)%(self.n_history+1)]
		
		self.next_weights=self.weights[(self.time_counter+1)%(self.n_history+1)]
		self.current_weights=self.weights[self.time_counter%(self.n_history+1)]
		self.previous_weights=self.weights[(self.time_counter-1)%(self.n_history+1)]
		
		self.proposal_samples=None
		self.log_proposal_probs=None
		
		self.true_log_transition_probs=None
		self.true_log_observation_probs=None
		
		self.perform_inference=None
		self.resample=None
		self.sample_joint=None
		
		
		
		ess=self.compute_ESS()
		self.get_ESS=theano.function([],ess)
		
		
	
	
	def recompile(self):
		'''This function compiles each of the theano functions that might
		change following a change of the model. '''
		
		data=T.fvector()
		samp_updates=self.sample_update(data)
		self.perform_inference=theano.function([data],updates=samp_updates,allow_input_downcast=True)
		
		res_updates=self.resample_update()
		self.resample=theano.function([],updates=res_updates)
		
		nsamps=T.lscalar()
		joint_samples, joint_updates=self.sample_from_joint(nsamps)
		self.sample_joint=theano.function([nsamps],joint_samples,updates=joint_updates)
		
		return
	
	
	def set_proposal(self, proposal_samples, log_proposal_probs):
		
		self.proposal_samples=proposal_samples
		self.log_proposal_probs=log_proposal_probs
		return
	
	
	def set_true_log_transition_probs(self, true_log_transition_probs):
		
		self.true_log_transition_probs=true_log_transition_probs
		return
	
	
	def set_true_log_observation_probs(self, true_log_observation_probs):
		
		self.true_log_observation_probs=true_log_observation_probs
		return
	
	
	def sample_update(self, data):
		
		proposal_samples=self.proposal_samples
		log_proposal_probs=self.log_proposal_probs
		
		log_transition_probs=self.true_log_transition_probs(self.current_state, proposal_samples)
		
		log_observation_probs=self.true_log_observation_probs(proposal_samples, data.dimshuffle('x',0))
		
		log_unnorm_weights=log_transition_probs + log_observation_probs - log_proposal_probs
		
		unnorm_weights=T.exp(log_unnorm_weights)*self.current_weights
		
		weights=unnorm_weights/T.sum(unnorm_weights)
		
		updates=OrderedDict()
		
		updates[self.weights]=T.set_subtensor(self.next_weights, weights)
		
		updates[self.particles]=T.set_subtensor(self.next_state, proposal_samples)
		
		updates[self.time_counter]=self.time_counter+1
		
		return updates
	
	
	def compute_ESS(self):
		
		return 1.0/T.sum(self.current_weights**2)
	
	
	def resample_update(self):
		
		#shape: n_particles by n_particles
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),self.n_particles,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		updates=OrderedDict()
		updates[self.particles]=T.set_subtensor(self.current_state, self.current_state[idxs])
		updates[self.weights]=T.set_subtensor(self.current_weights, T.cast(T.ones_like(self.current_weights)/float(self.n_particles),'float32'))
		return updates
	
	
	def sample_step(self, future_samps, t, n_samples):
		
		particles_now=self.particles[(self.time_counter-t)%(self.n_history+1)]
		weights_now=self.weights[(self.time_counter-t)%(self.n_history+1)]
		
		#n_particles by n_samples
		rel_log_probs=self.true_log_transition_probs(particles_now, future_samps, all_pairs=True)
		
		unnorm_probs=T.exp(rel_log_probs)*weights_now.dimshuffle(0,'x')
		probs=unnorm_probs/T.sum(unnorm_probs, axis=0).dimshuffle('x',0)
		
		samps=self.theano_rng.multinomial(pvals=probs.T)
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		output_samples=particles_now[idxs]
		
		return [output_samples, t+1]
	
	
	def sample_from_joint(self, n_samples):
		
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),n_samples,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samps_t0=self.current_state[idxs]
		
		t0=T.as_tensor_variable(1)
		
		[samples, ts], updates = theano.scan(fn=self.sample_step,
											outputs_info=[samps_t0, t0],
											non_sequences=[n_samples],
											n_steps=self.n_history)
		
		return samples, updates




		
	
	
	
