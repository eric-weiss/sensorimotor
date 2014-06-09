import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class LinearGaussian():
	
	def __init__(self, input_dims, output_dims, params_fn=None):
		
		self.input_dims=input_dims
		self.output_dims=output_dims
		
		self.theano_rng=RandomStreams()
		
		if params_fn==None:
			init_M=np.random.randn(input_dims,output_dims).astype(np.float32)
			init_b=np.zeros(output_dims).astype(np.float32)
			init_log_stddev=np.zeros(output_dims).astype(np.float32)
		
		self.M=theano.shared(init_M)
		self.b=theano.shared(init_b)
		self.log_stddev=theano.shared(init_log_stddev)
		
		self.params=[self.M, self.b, self.log_stddev]
	
	
	def compute_conditional_means(self, input_state):
		
		#input_state should be shape (n_particles, input_dims)
		
		M_terms=T.dot(input_state, self.M)  #(n_particles, output_dims)
		
		conditional_means=M_terms+self.b.dimshuffle('x',0)
		
		return conditional_means
	
	
	def get_samples(self, input_states):
		'''
		input_state should be shape (n_particles, input_dims)
		
		returns (samples, relative log probabilities under proposal distrib)
		'''
		
		conditional_means=self.compute_conditional_means(input_states)
		
		n=self.theano_rng.normal(size=conditional_means.shape)
		
		samps=conditional_means + n*T.exp(self.log_stddev).dimshuffle('x',0)
		
		return samps, -0.5*T.sum(n**2, axis=1)
	
	
	def rel_log_prob(self, input_states, output_samples, all_pairs=False,
									include_params_in_Z=False):
		''' computes the relative log probability of input and output samples
			up to an additive constant that may or may not depend on the parameters,
			depending on the value of include_params_in_Z
		
		input_states shape: (n_particles, input_dims)
		output_samples shape: (n_output, output_dims)
		
		output of the function will have shape:
			if all_pairs=False: (n_particles)
					otherwise: (n_particles, n_output)
		'''
		
		conditional_means=self.compute_conditional_means(input_states)
		
		if all_pairs:
			#(n_particles, n_output, output_dims)
			diffs=conditional_means.dimshuffle(0,'x',1)-output_samples.dimshuffle('x',0,1)
			
			#(n_particles, n_output)
			quadratic_terms=-T.sum((0.5*T.exp(-2.0*self.log_stddev)).dimshuffle('x','x',0)*(diffs**2),axis=2)
			
			if include_params_in_Z:
				Z_terms=-T.sum(self.log_stddev).dimshuffle('x','x')
			else:
				Z_terms=0
			
		else:
			#(n_particles, output_dims)
			diffs=conditional_means-output_samples
			
			#(n_particles)
			quadratic_terms=-T.sum((0.5*T.exp(-2.0*self.log_stddev)).dimshuffle('x',0)*(diffs**2),axis=1)
			
			if include_params_in_Z:
				Z_terms=-T.sum(self.log_stddev).dimshuffle('x')
			else:
				Z_terms=0
		
		return quadratic_terms + Z_terms


class MixedLinearGaussian():
	
	def __init__(self, input_dims, output_dims, n_components, params_fn=None):
		
		self.input_dims=input_dims
		self.output_dims=output_dims
		self.n_components=n_components
		
		self.theano_rng=RandomStreams()
		
		if params_fn==None:
			init_centers=np.random.randn(n_components,input_dims).astype(np.float32)
			init_biases=(np.zeros((n_components))-1.0).astype(np.float32)
			init_spreads=np.tile(0.1*np.eye(input_dims),(n_components,1,1)).astype(np.float32)
			#init_Ms=np.tile(np.zeros((input_dims,output_dims)),(n_components,1,1)).astype(np.float32)
			init_Ms=np.random.randn(n_components,input_dims,output_dims).astype(np.float32)
			init_bs=np.zeros((n_components,output_dims)).astype(np.float32)
			#init_bs=(np.random.randn(n_components,output_dims)*1).astype(np.float32)
			init_log_stddev=(np.zeros((output_dims))-0.0).astype(np.float32)
		
		self.centers=theano.shared(init_centers)
		self.biases=theano.shared(init_biases)
		self.spreads=theano.shared(init_spreads)
		self.Ms=theano.shared(init_Ms)
		self.bs=theano.shared(init_bs)
		self.log_stddev=theano.shared(init_log_stddev)
		
		self.params=[self.centers, self.biases, self.spreads, self.Ms, self.bs, self.log_stddev]
		
	
	def compute_conditional_means(self, input_state):
		
		#input_state should be shape (n_particles, input_dims)
		
		#diffs shape is (n_particles, n_components, input_dims)
		diffs=input_state.dimshuffle(0,'x',1)-self.centers.dimshuffle('x',0,1)
		
		#diffs_dot_spreads shape is also (n_particles, n_components, input_dims)
		diffs_dot_spreads=T.sum(diffs.dimshuffle(0,1,'x',2)*self.spreads.dimshuffle('x',0,1,2),axis=3)
		
		exp_terms=-T.sum(diffs_dot_spreads**2,axis=2) + self.biases.dimshuffle('x',0) #(n_particles, n_components)
		
		weights_unnorm=T.exp(exp_terms)
		
		weights=weights_unnorm/(T.sum(weights_unnorm,axis=1).dimshuffle(0,'x'))
		
		Ms_total=T.sum(weights.dimshuffle(0,1,'x','x')*self.Ms.dimshuffle('x',0,1,2),axis=1) #(n_particles, input_dims, output_dims)
		
		b_terms=T.sum(weights.dimshuffle(0,1,'x')*self.bs.dimshuffle('x',0,1),axis=1) #(n_particles, output_dims)
		
		M_terms=T.sum(Ms_total*input_state.dimshuffle(0,1,'x'),axis=1)  #(n_particles, output_dims)
		
		conditional_means=M_terms+b_terms
		
		return conditional_means
	
	
	def get_samples(self, input_states):
		'''
		input_state should be shape (n_particles, input_dims)
		
		returns (samples, relative log probabilities under proposal distrib)
		'''
		
		conditional_means=self.compute_conditional_means(input_states)
		
		n=self.theano_rng.normal(size=conditional_means.shape)
		
		samps=conditional_means + n*T.exp(self.log_stddev).dimshuffle('x',0)
		
		return samps, -0.5*T.sum(n**2, axis=1)
	
	
	def rel_log_prob(self, input_states, output_samples, all_pairs=False,
									include_params_in_Z=False):
		''' computes the relative log probability of input and output samples
			up to an additive constant that may or may not depend on the parameters,
			depending on the value of include_params_in_Z
		
		input_states shape: (n_particles, input_dims)
		output_samples shape: (n_output, output_dims)
		
		output of the function will have shape:
			if all_pairs=False: (n_particles)
					otherwise: (n_particles, n_output)
		'''
		
		conditional_means=self.compute_conditional_means(input_states)
		
		if all_pairs:
			#(n_particles, n_output, output_dims)
			diffs=conditional_means.dimshuffle(0,'x',1)-output_samples.dimshuffle('x',0,1)
			
			#(n_particles, n_output)
			quadratic_terms=-T.sum((0.5*T.exp(-2.0*self.log_stddev)).dimshuffle('x','x',0)*(diffs**2),axis=2)
			
			if include_params_in_Z:
				Z_terms=-T.sum(self.log_stddev).dimshuffle('x','x')
			else:
				Z_terms=0
			
		else:
			#(n_particles, output_dims)
			diffs=conditional_means-output_samples
			
			#(n_particles)
			quadratic_terms=-T.sum((0.5*T.exp(-2.0*self.log_stddev)).dimshuffle('x',0)*(diffs**2),axis=1)
			
			if include_params_in_Z:
				Z_terms=-T.sum(self.log_stddev).dimshuffle('x')
			else:
				Z_terms=0
		
		return quadratic_terms + Z_terms

