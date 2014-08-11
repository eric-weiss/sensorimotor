import numpy as np
import cPickle as cp
import theano
import theano.tensor as T
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams

class ParticleFilter():
	''' Implements particle filtering and smoothing for Markov Chains
	 with arbitrary proposal/true distributions '''
	
	def __init__(self, transition_model, observation_model, n_particles, observation_input=None, n_history=1):
		
		self.transition_model=transition_model
		self.observation_model=observation_model
		self.data_dims=observation_model.output_dims
		self.state_dims=transition_model.output_dims
		self.n_particles=n_particles
		self.n_history=n_history
		
		#this is used to keep track of what set of particles corresponds
		#to the previous point in time
		self.time_counter=theano.shared(0)
		
		self.theano_rng=RandomStreams()
		
		#init_particles=np.zeros((n_history+1, n_particles, self.state_dims)).astype(np.float32)
		init_particles=np.random.randn(n_history+1, n_particles, self.state_dims).astype(np.float32)
		init_weights=(np.ones((n_history+1, n_particles))/float(n_particles)).astype(np.float32)
		
		self.particles=theano.shared(init_particles)
		self.weights=theano.shared(init_weights)
		
		self.next_state=self.particles[(self.time_counter+1)%(self.n_history+1)]
		self.current_state=self.particles[self.time_counter%(self.n_history+1)]
		self.previous_state=self.particles[(self.time_counter-1)%(self.n_history+1)]
		
		self.next_weights=self.weights[(self.time_counter+1)%(self.n_history+1)]
		self.current_weights=self.weights[self.time_counter%(self.n_history+1)]
		self.previous_weights=self.weights[(self.time_counter-1)%(self.n_history+1)]
		
		self.proposal_distrib=None
		
		self.true_log_transition_probs=self.transition_model.rel_log_prob
		self.true_log_observation_probs=self.observation_model.rel_log_prob
		
		self.perform_inference=None
		self.resample=None
		self.sample_joint=None
		
		self.observation_input=observation_input
		
		ess=self.compute_ESS()
		self.get_ESS=theano.function([],ess)
		
		n_samps=T.lscalar()
		n_T=T.lscalar()
		data_samples, state_samples, init_state_samples, data_sample_updates=self.sample_future(n_samps,n_T)
		self.sample_from_future=theano.function([n_samps, n_T],[data_samples,state_samples,init_state_samples],updates=data_sample_updates)
		
		self.get_current_particles=theano.function([],self.current_state)
		self.get_current_weights=theano.function([],self.current_weights)
		
	
	def recompile(self):
		'''This function compiles each of the theano functions that might
		change following a change of the model. '''
		
		samp_updates=self.sample_update(self.observation_input)
		self.perform_inference=theano.function([],updates=samp_updates)
		
		res_updates=self.resample_update()
		self.resample=theano.function([],updates=res_updates)
		
		nsamps=T.lscalar()
		joint_samples, joint_updates=self.sample_from_joint(nsamps)
		self.sample_joint=theano.function([nsamps],joint_samples,updates=joint_updates)
		
		new_ess, stddevhist, esshist, sr_updates=self.sequential_resample()
		self.perform_sequential_resampling=theano.function([],[new_ess,stddevhist,esshist],updates=sr_updates)
		
		csamps=self.sample_current(nsamps)
		self.sample_current_state=theano.function([nsamps],csamps)
		
		psamps=self.sample_prev(nsamps)
		self.sample_previous_state=theano.function([nsamps],psamps)
		
		return
	
	
	def set_proposal(self, proposal_distrib):
		
		self.proposal_distrib=proposal_distrib
		
		return
	
	
	def set_true_log_transition_probs(self, true_log_transition_probs):
		
		self.true_log_transition_probs=true_log_transition_probs
		return
	
	
	def set_true_log_observation_probs(self, true_log_observation_probs):
		
		self.true_log_observation_probs=true_log_observation_probs
		return
	
	
	def sample_update(self, data):
		
		proposal_samples, log_proposal_probs=self.proposal_distrib
		
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
	
	
	def sample_from_joint(self, n_samples, output_2D=False):
		'''Samples from the joint posterior P(s_t-n_history:s_t | observations)
		n_samples: the number of samples to draw
		
		Returns an array with shape (n_history+1, n_samples, state_dims),
		where array[-1] corresponds to the current time.
		'''
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),n_samples,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samps_t0=self.current_state[idxs]
		
		t0=T.as_tensor_variable(1)
		
		[samples, ts], updates = theano.scan(fn=self.sample_step,
											outputs_info=[samps_t0, t0],
											non_sequences=[n_samples],
											n_steps=self.n_history)
		
		#the variable "samples" that results from the scan is time-flipped
		#in the sense that samples[0] corresponds to the most recent point
		#in time, and higher indices correspond to points in the past.
		#I will stick to the convention that for any collection of points in 
		#time, [-1] will index the most recent time, and [0] will index
		#the point farthest in the past. So, the first axis of "samples" 
		#needs to be flipped.
		flip_idxs=T.cast(-T.arange(self.n_history)+self.n_history-1,'int64')
		samples=T.concatenate([samples[flip_idxs], samps_t0.dimshuffle('x',0,1)], axis=0)
		
		if output_2D:
			samples=T.reshape(samples, ((self.n_history+1)*n_samples, self.state_dims))
		
		return samples, updates
	
	
	def sample_future(self, n_samples, n_T):
		'''Samples from the "future" data distribution: 
				P(s_t+1,...s_t+n_T, x_t+1,...x_t+n_T | s_t)
		
		n_samples: number of samples to draw
		n_T: the number of (future) time points to sample from
		
		Returns three arrays. The first two have shapes 
		(n_T, n_samples, data_dims) and
		(n_T, n_samples, state_dims),
		corresponding to samples of future observations and states,
		and the third having size (n_samples,state_dims),
		corresponding to the "initial" samples taken from the current
		state distribution.
		'''
		
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),n_samples,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samps_t0=self.current_state[idxs]
		
		state_samples, updates = theano.scan(fn=self.transition_model.get_samples_noprobs,
											outputs_info=[samps_t0],
											n_steps=n_T)
		
		data_samples=self.observation_model.get_samples_noprobs(state_samples)
		
		return data_samples, state_samples, samps_t0, updates
	
	
	def sample_model(self, n_samples, n_T):
		'''Samples from the "future" data distribution: 
				P(s_t+1,...s_t+n_T, x_t+1,...x_t+n_T | s_t)
		
		n_samples: number of samples to draw
		n_T: the number of (future) time points to sample from
		
		Returns three arrays. The first two have shapes 
		(n_T, n_samples, data_dims) and
		(n_T, n_samples, state_dims),
		corresponding to samples of future observations and states,
		and the third having size (n_samples,state_dims),
		corresponding to the "initial" samples taken from the current
		state distribution.
		'''
		
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),n_samples,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samps_t0=self.current_state[idxs]
		
		state_samples, updates = theano.scan(fn=self.transition_model.get_samples_noprobs,
											outputs_info=[samps_t0],
											n_steps=n_T)
		
		data_sample=self.observation_model.get_samples_noprobs(state_samples[-1])
		
		return data_sample, state_samples[-1], state_samples[-2], updates
	
	
	def sr_step(self, samples, ess, stddev, decay):
		proposal_samples=self.theano_rng.normal(size=samples.shape)*stddev.dimshuffle('x',0)+samples
		diffs=proposal_samples.dimshuffle(0,'x',1)-samples.dimshuffle('x',0,1)
		log_proposal_probs=T.log(T.sum(T.exp(-T.sum((1.0/(2.0*stddev**2)).dimshuffle('x','x',0)*diffs**2,axis=2)),axis=1))
		log_transition_probs=self.true_log_transition_probs(self.previous_state, proposal_samples,all_pairs=True)
		log_transition_probs=T.log(T.dot(T.exp(log_transition_probs).T,self.previous_weights))
		log_observation_probs=self.true_log_observation_probs(proposal_samples, self.observation_input.dimshuffle('x',0))
		log_unnorm_weights=log_transition_probs + log_observation_probs - log_proposal_probs
		unnorm_weights=T.exp(log_unnorm_weights)
		weights=unnorm_weights/T.sum(unnorm_weights)
		
		new_ess=1.0/T.sum(weights**2)
		
		#Resampling
		msamps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(weights.dimshuffle('x',0),samples.shape[0],axis=0))
		idxs=T.cast(T.dot(msamps, T.arange(samples.shape[0])),'int64')
		new_samples=T.cast(proposal_samples[idxs],'float32')
		
		sampmean=T.dot(proposal_samples.T, weights)
		sampvar=T.dot(((proposal_samples-sampmean.dimshuffle('x',0))**2).T,weights)
		#propmean=T.mean(proposal_samples, axis=0)
		#propvar=T.mean((proposal_samples-propmean.dimshuffle('x',0))**2,axis=0)
		#new_stddev=stddev*T.clip(T.exp(decay*(1.0-propvar/sampvar)),0.5,2.0)
		new_stddev=stddev*T.clip(T.exp(decay*(1.0-stddev**2/sampvar)),0.5,2.0)
		return [new_samples, T.cast(new_ess,'float32'), new_stddev], theano.scan_module.until(new_ess>100)
	
	
	def sequential_resample(self, nsamps=200, init_stddev=4.0, max_steps=120, stddev_decay=0.1):
		'''Repeatedly resamples and then samples from a proposal distribution
		constructed from the current samples. Should be used when the main
		proposal distribution is poor or whenever the ESS is poor.
		'''
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),nsamps,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		init_particles=self.current_state[idxs]
		
		essT=T.as_tensor_variable(np.asarray(0.0,dtype='float32'))
		stddevT=T.as_tensor_variable(np.asarray(init_stddev*np.ones(self.state_dims),dtype='float32'))
		decayT=T.as_tensor_variable(np.asarray(stddev_decay,dtype='float32'))
		
		[samphist, esshist, stddevhist], updates = theano.scan(fn=self.sr_step,
				outputs_info=[init_particles, essT, stddevT],
				non_sequences=decayT,
				n_steps=max_steps)
		
		end_samples=samphist[-1]
		end_stddev=stddevhist[-1]
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat((T.ones_like(T.arange(nsamps))/nsamps).dimshuffle('x',0),self.n_particles,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(nsamps)),'int64')
		#means=end_samples[idxs]
		means=end_samples
		proposal_samples=self.theano_rng.normal(size=(self.n_particles, self.state_dims))*end_stddev.dimshuffle('x',0)+means
		diffs=proposal_samples.dimshuffle(0,'x',1)-means.dimshuffle('x',0,1)
		log_proposal_probs=T.log(T.sum(T.exp(-T.sum((1.0/(2.0*end_stddev**2)).dimshuffle('x','x',0)*diffs**2,axis=2)),axis=1))
		log_transition_probs=self.true_log_transition_probs(self.previous_state, proposal_samples,all_pairs=True)
		log_transition_probs=T.log(T.dot(T.exp(log_transition_probs).T,self.previous_weights))
		log_observation_probs=self.true_log_observation_probs(proposal_samples, self.observation_input.dimshuffle('x',0))
		log_unnorm_weights=log_transition_probs + log_observation_probs - log_proposal_probs
		unnorm_weights=T.exp(log_unnorm_weights)
		weights=unnorm_weights/T.sum(unnorm_weights)
		
		updates[self.particles]=T.set_subtensor(self.current_state, proposal_samples)
		updates[self.weights]=T.set_subtensor(self.current_weights, weights)
		return 1.0/T.sum(weights**2), stddevhist, esshist, updates
	
	
	def sample_current(self, nsamps):
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.current_weights.dimshuffle('x',0),nsamps,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samples=self.current_state[idxs]
		return samples
	
	
	def sample_prev(self, nsamps):
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.previous_weights.dimshuffle('x',0),nsamps,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		samples=self.previous_state[idxs]
		return samples


class ImportanceSampler():
	'''Implements importance sampling/resampling'''
	
	def __init__(self, ndims, n_particles, true_log_probs, proposal_func=None):
		'''
		true_log_probs: a function that returns the true relative log probabilities
		proposal_func: a function that returns (samples, relative_log_probabilities)
		n_particles: the number of particles to use
		'''
		self.true_log_probs=true_log_probs
		self.proposal_func=proposal_func
		self.n_particles=n_particles
		self.ndims=ndims
		
		init_particles=np.zeros((n_particles, self.ndims))
		init_weights=np.ones(n_particles)/float(n_particles)
		
		self.particles=theano.shared(init_particles.astype(np.float32))
		self.weights=theano.shared(init_weights.astype(np.float32))
		
		self.theano_rng=RandomStreams()
		
		self.get_ESS=None
		self.perform_resampling=None
		self.perform_sampling=None
	
	
	def set_proposal_func(self, proposal_func):
		'''You might need to use this if you want to make the proposal
		function depend on the current particles'''
		self.proposal_func=proposal_func
		return
	
	
	def sample_reweight(self):
		'''Samples new particles and reweights them'''
		samples, prop_log_probs = self.proposal_func()
		true_log_probs=self.true_log_probs(samples)
		diffs=true_log_probs-prop_log_probs
		weights_unnorm=T.exp(diffs)
		weights=weights_unnorm/T.sum(weights_unnorm)
		updates=OrderedDict()
		updates[self.weights]=T.cast(weights,'float32')
		updates[self.particles]=T.cast(samples,'float32')
		return updates
	
	
	def compute_ESS(self):
		'''Returns the effective sample size'''
		return 1.0/T.sum(self.weights**2)
	
	
	def resample(self):
		'''Resamples using the current weights'''
		samps=self.theano_rng.multinomial(pvals=T.extra_ops.repeat(self.weights.dimshuffle('x',0),self.n_particles,axis=0))
		idxs=T.cast(T.dot(samps, T.arange(self.n_particles)),'int64')
		updates=OrderedDict()
		updates[self.particles]=self.particles[idxs]
		updates[self.weights]=T.cast(T.ones_like(self.weights)/float(self.n_particles),'float32')
		return updates
	
	
	def compile(self):
		'''Compiles the ESS, resampling, and sampling functions'''
		ess=self.compute_ESS()
		self.get_ESS=theano.function([],ess)
		resample_updates=self.resample()
		self.perform_resampling=theano.function([],updates=resample_updates)
		sample_updates=self.sample_reweight()
		self.perform_sampling=theano.function([],updates=sample_updates)
		return
		
		
		
	
	
	
