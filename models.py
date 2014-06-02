import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Mixed_Linear_Gaussian():
	
	def __init__(self, input_dims, output_dims, n_components, params_fn=None):
		
		self.input_dims=input_dims
		self.output_dims=output_dims
		self.n_components=n_components
		
		self.theano_rng=RandomStreams()
		
		if params_fn==None:
			init_centers=np.random.randn(n_components,input_dims).astype(np.float32)
			init_biases=np.zeros((n_components)).astype(np.float32)
			init_spreads=np.tile(np.eye(input_dims),(n_components,1,1)).astype(np.float32)
			init_Ms=np.tile(np.zeros((input_dims,output_dims)),(n_components,1,1)).astype(np.float32)
			init_bs=np.zeros((n_components,output_dims)).astype(np.float32)
			init_log_stddev=np.zeros((output_dims)).astype(np.float32)
		
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
		
		#input_state should be shape (n_particles, input_dims)
		
		conditional_means=self.compute_conditional_means(input_states)
		
		n=self.theano_rng.normal(size=conditional_means.shape)
		
		samps=conditional_means + n*T.exp(self.log_stddev).dimshuffle('x',0)
		
		return samps
	
	
	def compute_log_likelihood(self, input_states, output_samples):
		
		#input_states: (n_particles, input_dims)
		#output_samples: (n_samps, output_dims)
		
		#output of the function will be (n_particles, n_samps)
		
		conditional_means=self.compute_conditional_means(input_states)
		
		#(n_particles, n_samps, output_dims)
		diffs=conditional_means.dimshuffle(0,'x',1)-output_samples.dimshuffle('x',0,1)
		
		#(n_particles, n_samps)
		quadratic_terms=-T.sum((0.5*T.exp(-2.0*self.log_stddev)).dimshuffle('x','x',0)*(diffs**2),axis=2)
		
		Z_terms=-T.sum(self.log_stddev)
		
		return quadratic_terms + Z_terms.dimshuffle('x','x')


s=T.fmatrix()
y=T.fmatrix()
model=Mixed_Linear_Gaussian(3,5,7)
out=model.compute_log_likelihood(s,y)

test=theano.function([s,y],out,allow_input_downcast=True)

a=np.random.randn(8,3)
b=np.random.randn(7,5)

c=test(a,b)

print c.shape
